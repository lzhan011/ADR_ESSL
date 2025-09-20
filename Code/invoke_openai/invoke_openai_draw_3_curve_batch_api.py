import os, re, json, time
from tqdm import tqdm
from statistics import mean, median
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np

# =========================
# 目录/实验维度
# =========================
INPUT_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/cnf_results_openai_o1'
INPUT_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/cnf_results_openai_gpt-4o'
ALPHA_VALUES = list(np.arange(3.0, 6.0, 0.5))  # 仅用于聚合和画图；文件内 alpha 仍以文件名为准

# =========================
# 模型/Batch 配置
# =========================
MODEL = 'gpt-4.1'                  # 可改 'gpt-5' / 'o1' 等
OUTPUT_DIR = 'draw_deepseek_cnf_alpha_3_6_N_75_openai_prediction_' + MODEL

MAX_OUTPUT_TOKENS = 360           # 三行 + (可选) assignment 足够；越小越省排队额度
REASONING_EFFORT  = 'low'        # 'low' | 'medium' | 'high'（仅 /v1/responses 有效）
WRITE_FILE_CAP    = 10            # 本次处理的最大文件数；None=不限制
POLL_INTERVAL     = 30            # 轮询 batch 状态的秒数

# 平台“已排队 token”上限≈90k；给出更保守预算，仍会在失败时自动再细分
BATCH_TOKEN_LIMIT  = 90000
BATCH_TOKEN_BUDGET = 20000        # 单个 JSONL 预算（更小更稳）
FALLBACK_MAX_LINES = 6            # 失败后按行数细分的粒度
RETRY_MAX_TRIES    = 1
RETRY_WAIT_SEC     = 90

# =========================
# 工具函数
# =========================
def read_dimacs(filepath):
    clauses = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith(('c','p')):
                continue
            parts = line.strip().split()
            if 'time' in line.lower() and 'seconds' in line.lower():
                break
            if parts and parts[-1] == '0' and len(parts) == 4:
                clauses.append(list(map(int, parts[:-1])))
    return clauses

def clauses_to_dimacs_text(clauses):
    return "\n".join(" ".join(map(str, c)) + " 0" for c in clauses)

def build_prompt_dimacs(dimacs_text):
    # 极简提示，节省 token；不展示推理过程，只要结构化三行（SAT/UNSAT + branches + conflicts），SAT 时再给 assignment

    prompt = f"""You are a SAT logic solver.  
            Please use a step-by-step method to solve the following 3-CNF formula.  

            Finally, output only the following three items, with no extra explanation::
            * Whether the formula is SATISFIABLE or UNSATISFIABLE  
            * Number of branches (i.e., decision points)
            * Number of conflicts (i.e., backtracking steps)
            If the formula is SATISFIABLE, please give me the value for each literals.

            The formula is:
            {dimacs_text}
            """

    return prompt


def org_has_busy_batches(client, endpoint):
    # 粗略判断组织是否还有进行中的批注入（无法精确到具体模型，但对 gpt-4o=chat completions 很有参考意义）
    try:
        batches = client.batches.list(limit=100)
    except Exception:
        return False
    busy_status = {"validating", "in_progress", "finalizing"}
    for b in getattr(batches, "data", []):
        if getattr(b, "endpoint", None) == endpoint and getattr(b, "status", None) in busy_status:
            return True
    return False


def parse_response_free_text(text):
    low = (text or "").lower()
    sat = "unsatisfiable" not in low
    m_b = re.search(r"branches.*?(\d+)", low)
    m_c = re.search(r"conflicts.*?(\d+)", low)
    b = int(m_b.group(1)) if m_b else 0
    c = int(m_c.group(1)) if m_c else 0
    return sat, b, c

def find_k_n_alpha(file_name):
    matches = re.findall(r"N(\d+)|L(\d+)|alpha([\d.]+)|inst(\d+)", file_name)
    N = L = alpha = inst_idx = None
    for n,l,a,i in matches:
        if n: N = int(n)
        if l: L = int(l)
        if a: alpha = float(a)
        if i: inst_idx = int(i)
    return N, L, alpha, inst_idx

def endpoint_for(model: str) -> str:
    # gpt-5 / o1 / o3 → Responses；其余走 Chat Completions（gpt-4o 等）
    return "/v1/responses" if model.startswith(("gpt-5", "o1", "o3")) else "/v1/chat/completions"

def body_for_chat(model: str, prompt: str) -> dict:
    # Chat Completions 端点；gpt-4o 常用此端点
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }

def body_for(model: str, prompt: str, endpoint: str) -> dict:
    # Responses 端点（支持 reasoning.effort）
    if endpoint == "/v1/responses":
        body = {
            "model": model,
            "instructions": (
                "You are a SAT solver. Do all internal reasoning silently. "
                "At the END print ONLY:\n"
                "SATISFIABLE or UNSATISFIABLE\n"
                "branches: <int>\n"
                "conflicts: <int>\n"
                "assignment: x1=T x2=F ... (ONLY if SAT)\n"
                "No extra words."
            ),
            "input": prompt,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": {"verbosity": "low", "format": {"type": "text"}},
            "tool_choice": "none",
        }
        if REASONING_EFFORT:
            body["reasoning"] = {"effort": REASONING_EFFORT}
        return body
    else:
        return body_for_chat(model, prompt)

def parse_batch_content_text(obj):
    # 从 batch 输出 jsonl 的每一行抽出纯文本
    resp = obj.get("response", {}) or {}
    body = resp.get("body", {}) or {}

    ot = body.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot

    if "choices" in body:  # chat.completions
        ch = body.get("choices") or []
        if ch:
            msg = ch[0].get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, str): return content
            if isinstance(content, list):
                buf = []
                for p in content:
                    if isinstance(p, dict) and p.get("text"): buf.append(p["text"])
                    elif isinstance(p, str): buf.append(p)
                if buf: return "\n".join(buf)

    if "output" in body:  # responses 分块
        texts = []
        for block in body.get("output", []):
            if not isinstance(block, dict): continue
            if block.get("type") == "message":
                for c in block.get("content", []) or []:
                    if isinstance(c, dict) and c.get("text"): texts.append(c["text"])
                    elif isinstance(c, str): texts.append(c)
        if texts: return "\n".join(texts).strip()

    try:
        return obj["response"]["choices"][0]["message"]["content"]
    except Exception:
        return None

def validate_jsonl(jsonl_path, expected_endpoint):
    seen, total = set(), 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            total += 1
            obj = json.loads(line)
            cid = obj.get("custom_id")
            url = obj.get("url")
            body = obj.get("body", {}) or {}
            if cid in seen: raise ValueError(f"duplicate custom_id at line {ln}: {cid}")
            seen.add(cid)
            if url != expected_endpoint:
                raise ValueError(f"url mismatch at line {ln}: {url} != {expected_endpoint}")
            if expected_endpoint == "/v1/responses":
                if "input" not in body or not body["input"]:
                    raise ValueError(f"missing input at line {ln}")
                if "max_output_tokens" not in body:
                    raise ValueError(f"missing max_output_tokens at line {ln}")
            else:
                if not body.get("messages"): raise ValueError(f"missing messages at line {ln}")
    print(f"[OK] JSONL validated: {jsonl_path} (lines={total})")

def print_batch_status(batch):
    print("== Batch ==")
    print("id:", getattr(batch, "id", None))
    print("status:", getattr(batch, "status", None))
    print("input_file_id:", getattr(batch, "input_file_id", None))
    print("output_file_id:", getattr(batch, "output_file_id", None))
    err = getattr(batch, "error_file_id", None) or getattr(batch, "errors", None)
    if err: print("errors:", err)

def submit_batch_and_wait(client, input_file_id, endpoint, window="24h"):
    batch = client.batches.create(input_file_id=input_file_id, endpoint=endpoint, completion_window=window)
    print_batch_status(batch)
    while batch.status in ("validating", "in_progress", "finalizing"):
        print("Batch status:", batch.status, f"… waiting {POLL_INTERVAL}s")
        time.sleep(POLL_INTERVAL)
        batch = client.batches.retrieve(batch.id)
        print_batch_status(batch)
    return batch

# ===== 令牌估算 & 分批 =====
def estimate_tokens(s: str) -> int:
    # 粗估：~3.2 字符 ≈ 1 token，宁可高估
    if not s:
        return 0
    return int(len(s) / 3.2) + 1

def build_batches_from_dir(client, model, input_dir, output_dir, cap=None):
    os.makedirs(output_dir, exist_ok=True)
    endpoint = endpoint_for(model)
    task_meta = {}
    batch_jsonls = []

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    files.sort()

    batch_idx = 0
    cur_tokens = 0
    cur_count  = 0
    def new_jsonl_path(i): return f"batch_tasks_{model}_{i}.jsonl"
    wf = open(new_jsonl_path(batch_idx), "w", encoding="utf-8")

    written_total = 0
    for fname in tqdm(files, desc="[scan inputs]"):
        if cap and written_total >= cap: break
        src = os.path.join(input_dir, fname)
        if not os.path.exists(src): continue

        out_path = os.path.join(output_dir, fname)

        # 已有且无 error → 跳过
        skip = False
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8", errors='ignore') as rf:
                for line in rf:
                    if "error" in line.lower():
                        skip = False
                        break
                else:
                    skip = True
        if skip: continue

        clauses = read_dimacs(src)
        if not clauses:
            print(f"[WARN] no clauses parsed: {src}")
            continue

        dimacs_text = clauses_to_dimacs_text(clauses)
        prompt = build_prompt_dimacs(dimacs_text)
        body = body_for(model, prompt, endpoint)

        # 估算本任务 enqueued tokens
        if endpoint == "/v1/responses":
            in_tokens = estimate_tokens(body.get("instructions","")) + estimate_tokens(body.get("input",""))
            out_tokens = int(body.get("max_output_tokens", MAX_OUTPUT_TOKENS))
        else:  # chat.completions
            messages = body.get("messages", [])
            in_tokens = sum(estimate_tokens(m.get("content","")) for m in messages)
            out_tokens = int(body.get("max_tokens", MAX_OUTPUT_TOKENS))

        need = in_tokens + out_tokens

        # 当前批放不下 → 关闭当前 jsonl，开启新批
        if (cur_tokens + need > BATCH_TOKEN_BUDGET) and (cur_count > 0):
            wf.close()
            batch_jsonls.append(new_jsonl_path(batch_idx))
            batch_idx += 1
            cur_tokens = 0
            cur_count  = 0
            wf = open(new_jsonl_path(batch_idx), "w", encoding="utf-8")

        custom_id = f"b{batch_idx}::{fname}"
        task_meta[custom_id] = {
            "file_name": fname,
            "src": src,
            "out": out_path,
            "clauses": clauses,
        }

        row = {"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body}
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")

        cur_tokens += need
        cur_count  += 1
        written_total += 1

    wf.close()
    batch_jsonls.append(new_jsonl_path(batch_idx))
    print(f"[BUILD] built {len(batch_jsonls)} batch file(s), total tasks={written_total}, last batch tokens≈{cur_tokens}")
    return batch_jsonls, task_meta, endpoint

# ---------- 失败后再细分 JSONL（按行数） ----------
def split_jsonl_by_lines(jsonl_path, max_lines=FALLBACK_MAX_LINES):
    parts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    if len(lines) <= max_lines:
        return [jsonl_path]
    base = os.path.splitext(jsonl_path)[0]
    chunk = []
    idx = 0
    for ln in lines:
        chunk.append(ln)
        if len(chunk) >= max_lines:
            out = f"{base}_part{idx}.jsonl"
            with open(out, "w", encoding="utf-8") as wf:
                wf.writelines(chunk)
            parts.append(out)
            chunk = []
            idx += 1
    if chunk:
        out = f"{base}_part{idx}.jsonl"
        with open(out, "w", encoding="utf-8") as wf:
            wf.writelines(chunk)
        parts.append(out)
    return parts

# ---------- 读取 batch 错误文件并识别是否为“排队上限” ----------
def get_batch_error_text(client, batch):
    err_id = getattr(batch, "error_file_id", None)
    if err_id:
        try:
            return client.files.content(err_id).text
        except Exception:
            pass
    errs = getattr(batch, "errors", None)
    try:
        return json.dumps(errs) if errs else ""
    except Exception:
        return str(errs) if errs else ""

def is_enqueued_limit_error(txt: str) -> bool:
    if not txt:
        return False
    low = txt.lower()
    return ("token_limit_exceeded" in low) or ("enqueued token limit" in low)

# =========================
# 写出单文件 + 聚合统计
# =========================
def write_one_output(content_text, meta):
    fname = meta["file_name"]
    out   = meta["out"]
    clauses = meta["clauses"]

    N, L, alpha, inst_idx = find_k_n_alpha(fname)
    sat, branches, conflicts = parse_response_free_text(content_text)

    with open(out, "w", encoding="utf-8") as f:
        f.write("c Random 3-SAT\n")
        f.write(f"c alpha={round(alpha,2) if alpha is not None else 'NA'}, N={N}, L={L}, instance={inst_idx if inst_idx is not None else 'NA'}\n")
        f.write(f"p cnf {N} {L}\n")
        for clause in clauses:
            f.write(" ".join(str(x) for x in clause) + " 0\n")
        f.write(f"\nc GPT solve time: {0.00:.2f} seconds\n\n")
        f.write((content_text or "").strip())

    return sat, branches, conflicts, alpha

def aggregate_stats(alpha_values, all_rows):
    a2b, a2sat = {}, {}
    for a, sat, b, _c in all_rows:
        if a is None: continue
        a_key = round(float(a), 2)  # 统一两位小数
        a2b.setdefault(a_key, []).append(b)
        if sat: a2sat[a_key] = a2sat.get(a_key, 0) + 1
    mean_b, median_b, prob_s, avg_t = [], [], [], []
    for a in alpha_values:
        a_key = round(float(a), 2)
        lst = a2b.get(a_key, [])
        if lst:
            mean_b.append(mean(lst))
            median_b.append(median(lst))
            prob_s.append(a2sat.get(a_key, 0) / len(lst))
        else:
            mean_b.append(0); median_b.append(0); prob_s.append(0)
        avg_t.append(0.0)
    return mean_b, median_b, prob_s, avg_t

def plot_results(alpha_values, mean_branches, median_branches, prob_sat, avg_times,
                 title="OpenAI (Batch) on Random 3-SAT",
                 save_path="batch_result.png"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    l1, = ax1.plot(alpha_values, mean_branches, label="Mean branches")
    l2, = ax1.plot(alpha_values, median_branches, '--', label="Median branches")
    ax1.set_xlabel("L / N")
    ax1.set_ylabel("Number of branches")
    ax2 = ax1.twinx()
    l3, = ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)")
    ax2.set_ylabel("Prob(sat)")
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    l4, = ax3.plot(alpha_values, avg_times, '-.', label="Avg solve time (s)")
    ax3.set_ylabel("Avg solve time (s)")
    lines = [l1,l2,l3,l4]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.title(title); plt.grid(True); plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    print(f"[PLOT] saved to {save_path}")

# =========================
# 按批依次提交；失败→读取错误→等待→细分→重试
# =========================
def run_batches_sequentially(client, batch_jsonls, task_meta, endpoint):
    rows = []  # (alpha, sat, branches, conflicts)
    for jsonl_path in batch_jsonls:
        tries = 0
        to_submit = [jsonl_path]
        while tries < RETRY_MAX_TRIES and to_submit:
            next_round = []
            for path in to_submit:
                # 基本校验
                validate_jsonl(path, endpoint)

                # 预探测：若同端点仍有在排队的批次，就先 sleep 再试
                checks = 0
                while org_has_busy_batches(client, endpoint) and checks < 10:
                    time.sleep(RETRY_WAIT_SEC)  # 与你的等待参数一致
                    checks += 1
                # 然后再走 upload+create batch

                # 上传+提交
                try:
                    upload = client.files.create(file=open(path, "rb"), purpose="batch")
                    batch  = submit_batch_and_wait(client, upload.id, endpoint, window="24h")

                    status = getattr(batch, "status", None)
                    out_id = getattr(batch, "output_file_id", None)

                    if status == "completed" and out_id:
                        # 正常完成：解析输出
                        out_text = client.files.content(out_id).text
                        for line in out_text.splitlines():
                            if not line.strip(): continue
                            obj = json.loads(line)
                            cid = obj.get("custom_id", "")
                            content = parse_batch_content_text(obj)
                            meta = task_meta.get(cid)
                            if not meta:
                                print(f"[WARN] unknown custom_id: {cid}")
                                continue
                            sat, b, c, alpha = write_one_output(content, meta)
                            rows.append((alpha, sat, b, c))
                        continue

                    # 非 completed：读取错误详情
                    err_txt = get_batch_error_text(client, batch)
                    if is_enqueued_limit_error(err_txt):
                        print(f"[WARN] {os.path.basename(path)}: enqueued limit. sleep {RETRY_WAIT_SEC}s → split smaller …")
                        time.sleep(RETRY_WAIT_SEC)
                        parts = split_jsonl_by_lines(path, max_lines=FALLBACK_MAX_LINES)
                        if len(parts) == 1:  # 仍是一份 → 再细分
                            parts = split_jsonl_by_lines(path, max_lines=max(4, FALLBACK_MAX_LINES//2))
                        next_round.extend(parts)
                    else:
                        raise RuntimeError(f"Batch failed. status={status}. detail={err_txt or 'N/A'}")

                except Exception as e:
                    msg = str(e)
                    if is_enqueued_limit_error(msg):
                        print(f"[WARN] {os.path.basename(path)}: enqueued limit (exception). sleep {RETRY_WAIT_SEC}s → split …")
                        time.sleep(RETRY_WAIT_SEC)
                        parts = split_jsonl_by_lines(path, max_lines=FALLBACK_MAX_LINES)
                        if len(parts) == 1:
                            parts = split_jsonl_by_lines(path, max_lines=max(4, FALLBACK_MAX_LINES//2))
                        next_round.extend(parts)
                    else:
                        raise  # 其他错误直接抛出

            tries += 1
            to_submit = next_round
            if to_submit:
                print(f"[INFO] retry round {tries}, pending parts: {len(to_submit)}")
        if to_submit:
            raise RuntimeError("多次重试后仍有批次未能提交（组织额度可能持续被占用）。")
    return rows

# =========================
# 主流程
# =========================
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    api_key = os.environ["OPENAI_API_KEY"]

    if not api_key:
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 构建多个批次（自动分包）
    batch_jsonls, task_meta, endpoint = build_batches_from_dir(
        client, MODEL, INPUT_DIR, OUTPUT_DIR, cap=WRITE_FILE_CAP
    )
    if not task_meta:
        print("[INFO] 没有需要提交的样本（可能都已存在且无 error）。退出。")
        return

    # 逐批提交并处理输出（失败会自动读取错误→等待→细分→重试）
    rows = run_batches_sequentially(client, batch_jsonls, task_meta, endpoint)

    # 汇总并画图
    mean_b, median_b, prob_s, avg_t = aggregate_stats(ALPHA_VALUES, rows)
    plot_results(ALPHA_VALUES, mean_b, median_b, prob_s, avg_t,
                 title=f"OpenAI (Batch) — {MODEL}",
                 save_path=f"batch_result_{MODEL}.png")

if __name__ == "__main__":
    main()
