"""
Batch API 版（/v1/responses 去掉 text.format；其余逻辑不变）
依赖：
  pip install --upgrade openai matplotlib tqdm
使用：
  export OPENAI_API_KEY=sk-xxxx
说明：
  - 保持原有输出文件格式/命名/统计与绘图。
  - Batch 无法得到每条请求用时 -> 文件中的 "c GPT solve time:" 与 avg_times 统一记 0.00。
  - 将 WRITE_FILE_CAP 调整为一次批次的最大条数；或改造为“每个 N 一个批次”。
"""

import os
import re
import json
import time
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI

# =========================
# 全局参数
# =========================
MODEL_LIST = ['gpt-5']
# N_LIST = [50, 60, 70]
N_LIST = [80, 120, 140]
K = 3
ALPHA_VALUES = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]
INSTANCES_PER_ALPHA = 10000
WRITE_FILE_CAP = len(N_LIST) * 70 + 10
# WRITE_FILE_CAP = 2
API_KEY = os.getenv("OPENAI_API_KEY") or "REPLACE_ME"
API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# 工具函数
# =========================
def cnf_to_prompt(clauses):
    lines = []
    for clause in clauses:
        parts = []
        for lit in clause:
            parts.append(f"x{lit}" if lit > 0 else f"¬x{-lit}")
        lines.append("(" + " ∨ ".join(parts) + ")")
    return "\n".join(lines)

def generate_k_sat(n_vars, n_clauses, k):
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = sample(range(1, n_vars + 1), k)
        clause = [var if randint(0, 1) else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses

def parse_response_free_text(text):
    text_low = text.lower()
    sat = "unsatisfiable" not in text_low
    branches = re.search(r"branches.*?(\d+)", text_low)
    conflicts = re.search(r"conflicts.*?(\d+)", text_low)
    b = int(branches.group(1)) if branches else 0
    c = int(conflicts.group(1)) if conflicts else 0
    return sat, b, c

def read_dimacs_from_txt_like(filepath):
    clauses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[-1] == '0' and len(parts) == 4 and re.fullmatch(r"[\d\-]", line[0]):
                clause = list(map(int, parts[:-1]))
                clauses.append(clause)
    return clauses

def build_prompt_bak(cnf_text):
    return f"""You are a SAT solver. Solve the following 3-CNF internally, but DO NOT show any steps.
Return exactly three lines in English, nothing else:
SATISFIABLE or UNSATISFIABLE
branches: <int>
conflicts: <int>

The formula is:
{cnf_text}
"""


def build_prompt(cnf_text):
    prompt = f"""You are a SAT logic solver.  
    Please use a step-by-step method to solve the following 3-CNF formula.  

    Finally, output only the following three items, with no extra explanation:
    * Whether the formula is SATISFIABLE or UNSATISFIABLE  
    * Number of branches (i.e., decision points)
    * Number of conflicts (i.e., backtracking steps)
    If the formula is SATISFIABLE, please give me the value for each literal.

    The formula is:
    {cnf_text}
    """

    return prompt


def parse_batch_content_text(obj):
    resp = obj.get("response", {}) or {}
    body = resp.get("body", {}) or {}

    # A) Responses API 的聚合文本
    ot = body.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot

    # B) Chat Completions
    if "choices" in body:
        choices = body.get("choices") or []
        if choices:
            msg = choices[0].get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for piece in content:
                    if isinstance(piece, dict):
                        t = piece.get("text")
                        if t:
                            parts.append(t)
                    elif isinstance(piece, str):
                        parts.append(piece)
                if parts:
                    return "\n".join(parts)

    # C) Responses 分块输出：message.content[].text
    if "output" in body:
        texts = []
        for block in body.get("output", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "message":
                for c in block.get("content", []) or []:
                    if isinstance(c, dict) and c.get("text"):
                        texts.append(c["text"])
                    elif isinstance(c, str):
                        texts.append(c)
            elif "content" in block:
                for c in block.get("content", []) or []:
                    if isinstance(c, dict) and c.get("text"):
                        texts.append(c["text"])
        if texts:
            return "\n".join(texts).strip()

    # D) 兜底
    try:
        return obj["response"]["choices"][0]["message"]["content"]
    except Exception:
        return None

def should_skip_existing_output(output_file, source_txt_path):
    if not os.path.exists(output_file):
        return False
    error_flag_exist = False
    with open(source_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'error' in line.lower():
                error_flag_exist = True
                break
    return not error_flag_exist

# =========================
# 自检 & 日志
# =========================
def validate_jsonl(jsonl_path, expected_endpoint, raise_on_error=True):
    seen_ids = set()
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"[JSONL ERROR] line {ln}: invalid JSON ({e})"
                if raise_on_error: raise ValueError(msg)
                else: print(msg); continue

            cid = obj.get("custom_id")
            url = obj.get("url")
            body = obj.get("body", {}) or {}

            if not cid:
                msg = f"[JSONL ERROR] line {ln}: missing custom_id"
                if raise_on_error: raise ValueError(msg)
                else: print(msg)
            if cid in seen_ids:
                msg = f"[JSONL ERROR] line {ln}: duplicate custom_id={cid}"
                if raise_on_error: raise ValueError(msg)
                else: print(msg)
            seen_ids.add(cid)

            if url != expected_endpoint:
                msg = f"[JSONL ERROR] line {ln}: url={url} != {expected_endpoint}"
                if raise_on_error: raise ValueError(msg)
                else: print(msg)

            if not body.get("model"):
                msg = f"[JSONL ERROR] line {ln}: body.model missing"
                if raise_on_error: raise ValueError(msg)
                else: print(msg)

            if expected_endpoint == "/v1/chat/completions":
                msgs = body.get("messages", [])
                if not msgs or not msgs[0].get("content"):
                    msg = f"[JSONL ERROR] line {ln}: messages[0].content missing"
                    if raise_on_error: raise ValueError(msg)
                    else: print(msg)
                constrained = str(body.get("model","")).startswith(("gpt-5", "o1", "o3"))
                if constrained and "max_completion_tokens" not in body:
                    msg = f"[JSONL ERROR] line {ln}: missing max_completion_tokens for {body.get('model')}"
                    if raise_on_error: raise ValueError(msg)
                    else: print(msg)
                if (not constrained) and "max_tokens" not in body:
                    msg = f"[JSONL ERROR] line {ln}: missing max_tokens for {body.get('model')}"
                    if raise_on_error: raise ValueError(msg)
                    else: print(msg)

            elif expected_endpoint == "/v1/responses":
                if "input" not in body or not body["input"]:
                    msg = f"[JSONL ERROR] line {ln}: input missing for /v1/responses"
                    if raise_on_error: raise ValueError(msg)
                    else: print(msg)
                if "max_output_tokens" not in body:
                    msg = f"[JSONL ERROR] line {ln}: max_output_tokens missing for /v1/responses"
                    if raise_on_error: raise ValueError(msg)
                    else: print(msg)

    print(f"[OK] JSONL validated: {jsonl_path}  (total lines={total})")

def print_batch_status(batch):
    print("== Batch ==")
    print("id:", getattr(batch, "id", None))
    print("status:", getattr(batch, "status", None))
    print("input_file_id:", getattr(batch, "input_file_id", None))
    print("output_file_id:", getattr(batch, "output_file_id", None))
    err_id = getattr(batch, "error_file_id", None) or getattr(batch, "errors", None)
    if err_id:
        print("error_file_id or errors:", err_id)

def dump_error_file_if_any(client, batch):
    err_id = getattr(batch, "error_file_id", None)
    if not err_id:
        return
    try:
        txt = client.files.content(err_id).text
        print("== Batch Error File BEGIN ==")
        for i, line in enumerate(txt.splitlines(), 1):
            print(line)
            if i >= 30:
                print("... (truncated)")
                break
        print("== Batch Error File END ==")
    except Exception as e:
        print("[WARN] cannot fetch error file:", e)

# =============== 端点/Body 适配 ===============
def endpoint_for(model: str) -> str:
    return "/v1/responses" if model.startswith(("gpt-5", "o1", "o3")) else "/v1/chat/completions"

def body_for_chat(model: str, prompt: str) -> dict:
    constrained = model.startswith(("gpt-5", "o1", "o3"))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if constrained:
        body["max_completion_tokens"] = 4096
    else:
        body["max_tokens"] = 4096
        body["temperature"] = 0
    return body



def body_for(model: str, prompt: str, endpoint: str) -> dict:
    REASONING_EFFORT = "low"  # "low" | "medium" | "high" | None
    MAX_OUTPUT_TOKENS = 4096  # 若仍出现只 reasoning，无文本，就再调大到 4096
    if endpoint == "/v1/responses":
        body = {
            "model": model,
            "instructions": (
                                "You are a SAT solver. Perform any internal reasoning silently.\n"
                "At the END you MUST print EXACTLY these lines and NOTHING ELSE:\n"
                "SATISFIABLE or UNSATISFIABLE\n"
                "branches: <int>\n"
                "conflicts: <int>\n"
                "assignment: x1=T x2=F x3=T ... xN=F\n"
                "\n"
                "Hard constraints:\n"
                "- Always print the first three lines.\n"
                "- Print the 'assignment:' line ONLY IF SATISFIABLE; omit it if UNSATISFIABLE.\n"
                "- The entire final answer MUST be <= 600 tokens and contain no blank lines or extra words.\n"
                "- Do NOT include any reasoning or explanations."
            ),
            "input": prompt,                   # 只放公式
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": {"verbosity": "low", "format": {"type": "text"}},
            "tool_choice": "none",
        }
        if REASONING_EFFORT:
            body["reasoning"] = {"effort": REASONING_EFFORT}
        return body
    else:
        return body_for_chat(model, prompt)



# =========================
# Batch 构建 / 提交 / 解析
# =========================
def build_batch_jsonl_for_model(client, model_selected, write_cap=WRITE_FILE_CAP):
    used_alpha_values = list(ALPHA_VALUES)
    task_meta = {}
    jsonl_path = f"batch_tasks_{model_selected}.jsonl"
    written = 0
    endpoint_used = endpoint_for(model_selected)

    with open(jsonl_path, "w", encoding="utf-8") as wf:
        for N in N_LIST:
            input_dir_o1 = f'unsat_cnf_low_alpha_N_{N}_openai_prediction_o1'
            output_dir = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_{model_selected}"
            os.makedirs(output_dir, exist_ok=True)

            for alpha in tqdm(ALPHA_VALUES, desc=f"[{model_selected}] scan N={N} alpha"):
                L = int(alpha * N)
                for inst_idx in range(1, INSTANCES_PER_ALPHA + 1):
                    src_name = f"cnf_k{K}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx}.txt"
                    source_txt_path = os.path.join(input_dir_o1, src_name)
                    if not os.path.exists(source_txt_path):
                        continue

                    output_file = os.path.join(output_dir, src_name)
                    # if should_skip_existing_output(output_file, source_txt_path):
                    #     continue

                    clauses = read_dimacs_from_txt_like(source_txt_path)
                    if not clauses:
                        print(f"[WARN] No clauses parsed from {source_txt_path}, skip.")
                        continue

                    cnf_text = cnf_to_prompt(clauses)
                    prompt = build_prompt(cnf_text)

                    custom_id = f"m_{model_selected}_N{N}_alpha{alpha}_L{L}_inst{inst_idx}"
                    task_meta[custom_id] = {
                        "model": model_selected,
                        "N": N,
                        "alpha": alpha,
                        "L": L,
                        "inst": inst_idx,
                        "clauses": clauses,
                        "output_file": output_file
                    }

                    row = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": endpoint_used,
                        "body": body_for(model_selected, prompt, endpoint_used)
                    }
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= write_cap:
                        break
                if written >= write_cap:
                    break
            if written >= write_cap:
                break

    print(f"[BUILD] {jsonl_path} built with {written} lines.")
    return jsonl_path, task_meta, used_alpha_values, endpoint_used

def submit_batch_and_wait(client, input_file_id, endpoint, window="24h"):
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=window
    )
    print_batch_status(batch)
    while batch.status in ("validating", "in_progress", "finalizing"):
        print("Batch status:", batch.status, "… waiting 30s")
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print_batch_status(batch)
    return batch

def write_txt_and_aggregate(cid, content_text, task_meta,
                            alpha_to_branches, alpha_to_sat_cnt, alpha_to_times):
    meta = task_meta.get(cid)
    if not meta or content_text is None:
        return

    N = meta["N"]
    alpha = float(meta["alpha"])
    L = meta["L"]
    inst_idx = meta["inst"]
    clauses = meta["clauses"]
    output_file = meta["output_file"]

    sat, branches, conflicts = parse_response_free_text(content_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("c Random 3-SAT\n")
        f.write(f"c alpha={round(alpha,2)}, N={N}, L={L}, instance={inst_idx}\n")
        f.write(f"p cnf {N} {L}\n")
        for clause in clauses:
            f.write(" ".join(str(x) for x in clause) + " 0\n")
        f.write(f"\nc GPT solve time: {0.00:.2f} seconds\n\n")
        f.write(content_text.strip() if content_text else "")

    quick_check_written_txt(output_file, N, L)

    alpha_to_branches.setdefault(alpha, []).append(branches)
    if sat:
        alpha_to_sat_cnt[alpha] = alpha_to_sat_cnt.get(alpha, 0) + 1
    alpha_to_times.setdefault(alpha, []).append(0.0)

def quick_check_written_txt(path, N, L):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().lower()
    if not re.search(rf"^p\s+cnf\s+{N}\s+{L}\s*$", data, flags=re.M):
        raise AssertionError(f"[TXT CHECK] missing or wrong 'p cnf {N} {L}' in {path}")
    if not re.search(r"^\s*-?\d+\s+-?\d+\s+-?\d+\s+0\s*$", data, flags=re.M):
        raise AssertionError(f"[TXT CHECK] no 3-literal clause lines ending with 0 in {path}")
    if ("satisfiable" not in data) and ("unsatisfiable" not in data):
        raise AssertionError(f"[TXT CHECK] missing 'satisfiable/unsatisfiable' in {path}")

def aggregate_to_arrays(alpha_values, alpha_to_branches, alpha_to_sat_cnt, alpha_to_times):
    mean_branches, median_branches, prob_sat, avg_times = [], [], [], []
    for a in alpha_values:
        a = float(a)
        blist = alpha_to_branches.get(a, [])
        tlist = alpha_to_times.get(a, [])
        sat_cnt = alpha_to_sat_cnt.get(a, 0)
        if blist:
            mean_branches.append(mean(blist))
            median_branches.append(median(blist))
            avg_times.append(mean(tlist) if tlist else 0.0)
            prob_sat.append(sat_cnt / len(blist))
        else:
            mean_branches.append(0)
            median_branches.append(0)
            avg_times.append(0)
            prob_sat.append(0)
    return mean_branches, median_branches, prob_sat, avg_times

def plot_results(alpha_values, mean_branches, median_branches, prob_sat, avg_times,
                 title="OpenAI (Batch) on Random 3-SAT (UNSAT set replay)",
                 save_path="batch_result.png"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, mean_branches, label="Mean branches", color="black")
    ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="black")
    line1, = ax1.plot(alpha_values, mean_branches, label="Mean branches", color="black")
    line2, = ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="black")
    ax1.set_xlabel("L / N")
    ax1.set_ylabel("Number of branches")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="blue")
    ax2.set_ylabel("Prob(sat)")
    line3, = ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="blue")
    ax2.legend(loc="lower left")

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(alpha_values, avg_times, '-.', label="Avg solve time (s)", color="gray")
    ax3.set_ylabel("Avg solve time (seconds)")
    line4, = ax3.plot(alpha_values, avg_times, '-.', label="Avg solve time (s)", color="gray")
    ax3.legend(loc="upper right")

    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] saved to {save_path}")

def run_for_model(client, model_selected):
    jsonl_path, task_meta, used_alpha_values, endpoint_used = build_batch_jsonl_for_model(client, model_selected)

    if not task_meta:
        print(f"[{model_selected}] 没有需要提交的样本（可能都已存在且无 error）。跳过。")
        zeros = [0] * len(used_alpha_values)
        return zeros, zeros, zeros, zeros

    validate_jsonl(jsonl_path, endpoint_used)

    upload = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = submit_batch_and_wait(client, upload.id, endpoint_used, window="24h")

    status = getattr(batch, "status", None)
    out_id = getattr(batch, "output_file_id", None)

    if status != "completed":
        print(f"[{model_selected}] Batch 未完成，status={status}")
        dump_error_file_if_any(client, batch)
        raise RuntimeError(f"[{model_selected}] Batch ended with status={status}")

    if not out_id:
        print(f"[{model_selected}] Batch 已完成但没有 output_file_id（可能 0 行或全失败）。")
        dump_error_file_if_any(client, batch)
        raise RuntimeError(f"[{model_selected}] Completed with no output_file_id")

    out_text = client.files.content(out_id).text

    alpha_to_branches, alpha_to_sat_cnt, alpha_to_times = {}, {}, {}
    for line in out_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id", "")
        content_text = parse_batch_content_text(obj)
        if content_text is None or content_text.strip() == "":
            body = (obj.get("response", {}) or {}).get("body", {}) or {}
            print(f"[PARSE-OUTPUT WARN] custom_id={cid} no text; saving debug JSON")
            # >>> 新增：保存原始响应，便于排查
            debug_dir = "batch_debug_raw"
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{cid}.debug.json"), "w", encoding="utf-8") as df:
                json.dump(body, df, ensure_ascii=False, indent=2)
            continue
        write_txt_and_aggregate(cid, content_text, task_meta,
                                alpha_to_branches, alpha_to_sat_cnt, alpha_to_times)

    return aggregate_to_arrays(used_alpha_values, alpha_to_branches, alpha_to_sat_cnt, alpha_to_times)

# =========================
# 主程序
# =========================
def main():
    if API_KEY == "REPLACE_ME":
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")
    client = OpenAI(api_key=API_KEY)

    last_mean, last_median, last_prob, last_avg = None, None, None, None
    for model in MODEL_LIST:
        print(f"\n==== Running Batch for model: {model} ====")
        mean_b, median_b, prob_s, avg_t = run_for_model(client, model)
        last_mean, last_median, last_prob, last_avg = mean_b, median_b, prob_s, avg_t

    plot_results(ALPHA_VALUES, last_mean, last_median, last_prob, last_avg,
                 title=f"OpenAI (Batch) on Random 3-SAT (UNSAT set replay) — {MODEL_LIST[-1]}",
                 save_path=f"batch_result_{MODEL_LIST[-1]}.png")

if __name__ == "__main__":
    main()
