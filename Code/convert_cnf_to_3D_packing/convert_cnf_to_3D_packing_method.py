# --- add project root to sys.path ---
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # -> Satisfiability_Solvers
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Code.convert_cnf_to_vertex_cover.convert_cnf_to_vertex_cover_method_2 import  *
import time


from typing import List, Dict, Any
import json

# --------- Helpers for literals ---------
def literal_var(lit: str) -> str:
    """'x7' or '!x7' -> 'x7'"""
    return lit[1:] if lit.startswith('!') else lit

def literal_is_neg(lit: str) -> bool:
    """Return True iff the literal is negated (starts with '!')."""
    return lit.startswith('!')

# --------- Main: 3-CNF -> 3D packing (discrete grid) ----------
def build_3d_packing_from_3cnf(clauses_3: List[List[str]]) -> Dict[str, Any]:
    """
    Input:
        clauses_3: e.g. [["x1","!x2","x3"], ["!x1","x2","!x3"], ...]
    Output: a structured packing instance that explicitly lists container, objects, and constraints.

    Convention:
      - Container size = [n, 2, m] (X = #variables, Y = 2 layers for True/False, Z = #clauses)
      - Variable objects (rods): for each variable, two rods (True/False), size [1,1,m], choose exactly one
      - Clause objects (tokens): one token per clause, size [1,1,1]
        Allowed positions (allowed_slots) are derived from the clause's 3 literals:
          * x_i  -> (x=i, y=0, z=j)
          * !x_i -> (x=i, y=1, z=j)
      - Semantic constraints (for a solver/LLM to read):
          * variables_exactly_one: for each variable, choose exactly one rod (T or F)
          * tokens_land_on_selected_rods: a token must be placed on a cell covered by the chosen rod
    """
    # 1) Collect variables and assign x-axis indices 1..n (for readability)
    vars_sorted = sorted({literal_var(l) for C in clauses_3 for l in C},
                         key=lambda s: int(s[1:]))  # 'x7' -> 7
    var2x = {v: i+1 for i, v in enumerate(vars_sorted)}  # 1-based x indices
    n = len(vars_sorted)
    m = len(clauses_3)

    # 2) Build rods (variable objects)
    rods = []
    var_groups_exactly_one = []  # per-variable "choose exactly one" sets
    for v, x in var2x.items():
        rod_T_id = f"rod_T_{v}"      # y=0
        rod_F_id = f"rod_F_{v}"      # y=1
        rods.append({
            "id": rod_T_id,
            "type": "rod",
            "variable": v,
            "truth": True,
            "size": [1, 1, m],
            "anchor": {"x": x, "y": 0, "z": 0},  # spans all z
            "locked_orientation": True
        })
        rods.append({
            "id": rod_F_id,
            "type": "rod",
            "variable": v,
            "truth": False,
            "size": [1, 1, m],
            "anchor": {"x": x, "y": 1, "z": 0},
            "locked_orientation": True
        })
        var_groups_exactly_one.append({"exactly_one_of": [rod_T_id, rod_F_id]})

    # 3) Build tokens (clause objects) with allowed_slots
    tokens = []
    for j, clause in enumerate(clauses_3):
        tok_id = f"token_C{j}"
        allowed_slots = []
        for lit in clause:
            v = literal_var(lit)         # 'x3'
            x = var2x[v]                 # variable -> x coordinate
            y = 0 if not literal_is_neg(lit) else 1
            allowed_slots.append({"x": x, "y": y, "z": j})
        tokens.append({
            "id": tok_id,
            "type": "token",
            "clause_index": j,
            "size": [1, 1, 1],
            "allowed_slots": allowed_slots,
            # semantics: must be placed in one allowed slot AND that cell must be covered by the chosen rod
            "must_land_on_selected_rod": True
        })

    # 4) Assemble instance
    instance = {
        "kind": "grid_packing_3d_for_3sat_explicit",
        "container": {
            "size": [n, 2, m],     # [X, Y, Z]
            "rotations": False
        },
        "objects": {
            "rods": rods,
            "tokens": tokens
        },
        "constraints": {
            "variables_exactly_one": var_groups_exactly_one,
            "tokens_land_on_selected_rods": True
        },
        "legend": {
            "coords_meaning": "x=variable index (1..n), y in {0(True),1(False)}, z=clause index (0..m-1)",
            "semantics": [
                "For each variable choose exactly one: rod_T_xi or rod_F_xi.",
                "Each clause token must be placed on one of its allowed slots, and that cell must be on a chosen rod.",
                "Packing feasible ⇔ the 3-CNF is satisfiable."
            ]
        }
    }
    return instance



# ------------------ DIMACS reader (robust) ------------------
def read_dimacs(filepath):
    """
    Parse standard DIMACS CNF:
      c ...  (comments)
      p cnf <nvars> <nclauses>
      <l1> <l2> ... <lk> 0
    Returns: List[List[int]] e.g., [[1,-2,3], ...]
    """
    clauses = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('c') or s.startswith('p'):
                continue
            parts = s.split()
            cur = []
            for token in parts:
                try:
                    v = int(token)
                except ValueError:
                    cur = []
                    break
                if v == 0:
                    if cur:
                        clauses.append(cur)
                    cur = []
                else:
                    cur.append(v)
            # Handle lines without trailing 0 (rare, but some writers break across lines)
            if cur and parts[-1] != '0':
                # carry to next line: do nothing; next lines should complete until we see a 0
                # To keep this simple, assume well-formed inputs end each clause with 0
                pass
    return clauses

# ------------------ 3-CNF -> 3D Packing instance ------------------
def build_3d_packing_from_3cnf_bak(clauses_3):
    """
    Input: clauses_3 : List[List[str]] with each literal like 'x7' or '!x7'
    Output: instance dict for a grid-packing decision:
      - container: width=n_vars, height=2, depth=n_clauses, rotations=False
      - variables: for i = 1..n, rods (T at y=0; F at y=1)
      - clauses[j]: allowed slots (x=i, y in {0,1}, z=j) from its 3 literals
    Packing rule (for LLM): choose exactly one rod per variable (T or F);
    for each clause j, place 1 token at some allowed slot that lies on a chosen rod.
    """
    # collect variables
    vars_set = sorted({literal_var(l) for C in clauses_3 for l in C},
                      key=lambda s: int(s[1:]))  # 'x7' -> 7
    idx = {v: i+1 for i, v in enumerate(vars_set)}  # 1-based for readability
    n = len(vars_set)
    m = len(clauses_3)

    # slots per clause
    clause_slots = []
    for j, C in enumerate(clauses_3):
        slots = []
        for lit in C:
            v = literal_var(lit)
            i = idx[v]
            y = 0 if not literal_is_neg(lit) else 1
            slots.append([i, y, j])  # x=i, layer=y, depth=j
        clause_slots.append({"index": j, "slots": slots})

    instance = {
        "kind": "grid_packing_3d_for_3sat",
        "container": {"width": n, "height": 2, "depth": m, "rotations": False},
        "variables": [
            {
                "name": v,
                "index": i,
                "rod_true":  {"x": i, "y": 0, "z0": 0, "len_z": m},
                "rod_false": {"x": i, "y": 1, "z0": 0, "len_z": m}
            }
            for v, i in idx.items()
        ],
        "clauses": clause_slots,
        "note": "Place EXACTLY ONE rod (true or false) for each variable. "
                "For each clause j, place ONE token cube at one of its allowed slots; "
                "tokens must sit on an occupied rod cell. Feasible <=> the 3-CNF is satisfiable."
    }
    return instance




def save_3d_packing_instance(instance, out_dir, base_filename, suffix="_3dpack.json"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(base_filename)[0] + suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(instance, f, ensure_ascii=False, indent=2)
    return out_path

# ------------------ 3D Packing LLM prompt ------------------
import json

def make_3d_packing_prompt(instance, instance_name=None):
    # container.size = [X, Y, Z]
    X, Y, Z = instance["container"]["size"]
    header = (
        "You are an expert in discrete 3D packing and constraint solving.\n"
        f"- Container (grid): X = {X}, Y = {Y}, Z = {Z}; rotations are NOT allowed and orientations are locked.\n"
        "- Objects come in TWO types (no other types exist):\n"
        "  * Type-A (long rods): for each x-index in {1..X} there are EXACTLY TWO rods, anchored at y=0 and y=1, "
        "each with size [1,1,Z] and spanning cells (x=const, y∈{0,1}, z=0..Z-1). "
        "You must SELECT EXACTLY ONE rod per x-index (choose y=0 OR y=1), never both.\n"
        "  * Type-B (unit cubes, “tokens”): there is EXACTLY ONE Type-B object for each depth z=j (j=0..Z-1). "
        "Each Type-B object lists its allowed placements as coordinates in its `allowed_slots` field.\n"
        "- Valid placement rule:\n"
        "  * A Type-B placement at (x,y,z=j) is VALID ONLY if that cell is covered by a SELECTED Type-A rod at the same (x,y).\n"
        "  * You MUST choose coordinates ONLY from the `allowed_slots` provided; do not invent coordinates.\n"
        "- Feasibility means EVERY Type-B object can be placed validly under these rules.\n\n"
        "Return ONLY a ONE-LINE JSON object:\n"
        '  {"answer":"YES"|"NO",'
        ' "assignment":{"1":true/false,"2":true/false,...,"' + str(X) + '":true/false},'
        ' "tokens":[[x,y,z],...],'
        ' "explain":"<≤2 sentences>"}\n'
        "Where:\n"
        '  - `assignment` uses STRING KEYS for ALL x-indices 1..X; true ⇒ select the rod at y=0, false ⇒ select the rod at y=1.\n'
        "  - `tokens` must contain EXACTLY one coordinate for EACH depth layer z=0..Z-1, in ascending z order, "
        "and each coordinate must be one of that layer’s `allowed_slots`.\n"
        "If ANY Type-B object cannot be validated, return NO with empty assignment and tokens.\n"
        "Do not include any extra text before or after the one-line JSON.\n"
    )

    name_line = (f"Instance: {instance_name}\n" if instance_name else "")
    body = (
        f"{name_line}"
        "3D packing instance (JSON):\n"
        + json.dumps(instance, ensure_ascii=False)
        + "\n"
        "Task: Decide if the packing is feasible under the rules above.\n"
    )
    return header + "\n" + body


# ------------------ Parse LLM answer for 3D packing ------------------
def parse_llm_answer_json_packing(ans_text: str):
    """
    Expected one-line JSON:
      {"answer":"YES"|"NO","assignment":{"x1":true,...},"tokens":[[x,y,z],...], "explain":"..."}
    Returns: ok(bool), is_yes(bool|None), assignment(dict), tokens(list)
    """
    if not ans_text:
        return False, None, {}, []
    m = re.search(r"\{.*\}", ans_text, flags=re.DOTALL)
    s = m.group(0) if m else ans_text.strip()
    try:
        obj = json.loads(s)
    except Exception:
        return False, None, {}, []
    ans = str(obj.get("answer","")).strip().upper()
    if ans not in ("YES","NO"):
        return False, None, {}, []
    asg = obj.get("assignment", {}) if ans == "YES" else {}
    toks = obj.get("tokens", []) if ans == "YES" else []
    # normalize assignment keys (x1..xn) to bool
    norm_asg = {}
    try:
        for k,v in asg.items():
            if isinstance(v, bool):
                norm_asg[k] = v
            elif isinstance(v, str):
                vv = v.strip().lower()
                if vv in ("true","t","1","yes"): norm_asg[k] = True
                elif vv in ("false","f","0","no"): norm_asg[k] = False
                else: return False, None, {}, []
            else:
                return False, None, {}, []
    except Exception:
        return False, None, {}, []
    # tokens should be triples of ints
    try:
        toks2 = []
        for t in toks:
            if not (isinstance(t, list) and len(t)==3): return False, None, {}, []
            x,y,z = map(int, t)
            toks2.append([x,y,z])
    except Exception:
        return False, None, {}, []
    return True, (ans=="YES"), norm_asg, toks2

def normalize_assignment_keys_for_eval(asg: Dict[str, bool]) -> Dict[str, bool]:
    out = {}
    for k, v in (asg or {}).items():
        ks = str(k).strip()
        if ks.startswith('x'):
            out[ks] = bool(v)
        elif ks.isdigit():
            out['x' + ks] = bool(v)
        else:
            out[ks] = bool(v)
    return out


# ------------------ CNF check helpers ------------------
def eval_3cnf_assignment(clauses_3, asg_dict):
    """asg_dict: {'x1':True/False, ...}"""
    for C in clauses_3:
        sat = False
        for lit in C:
            var = literal_var(lit)
            val = asg_dict.get(var, None)
            if val is None:
                return False
            lit_true = val if not literal_is_neg(lit) else (not val)
            sat = sat or lit_true
        if not sat:
            return False
    return True

def verify_llm_assignment_against_cnf(clauses_3, asg_raw):
    """
    返回 (ok: bool, reason: str)
    ok=True 表示 assignment 能满足公式
    """
    asg = normalize_assignment_keys_for_eval(asg_raw or {})
    vars_in_formula = { literal_var(l) for C in clauses_3 for l in C }
    missing = [v for v in vars_in_formula if v not in asg]
    if missing:
        return False, f"assignment missing vars: {missing[:10]}{'...' if len(missing)>10 else ''}"
    # 逐子句检查
    ok = eval_3cnf_assignment(clauses_3, asg)
    return (ok, "ok" if ok else "violates at least one clause")


# ------------------ End-to-end: per-file 3D packing prompt/run ------------------
def process_one_file_3d_packing(filepath, packing_out_dir=None, model_selected=""):
    """
    - Read DIMACS CNF
    - Convert to 3-CNF strings
    - Build 3D packing instance JSON + save
    - Make prompt + save
    - Call LLM, save answer
    - Parse/validate against the CNF truth
    """
    clauses_int = read_dimacs(filepath)
    clauses_3 = int_clauses_to_str_3cnf(clauses_int)  # ensures 3-CNF

    # Build + save 3D packing
    if packing_out_dir is None:
        packing_out_dir = os.path.join(os.path.dirname(filepath), "packing")
    instances_dir = os.path.join(packing_out_dir, "instances")
    prompts_dir   = os.path.join(packing_out_dir, "prompts")
    answers_dir   = os.path.join(packing_out_dir, "answers")
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(answers_dir, exist_ok=True)

    inst = build_3d_packing_from_3cnf(clauses_3)
    base = os.path.basename(filepath)
    inst_path = save_3d_packing_instance(inst, instances_dir, base)

    # Prompt
    prompt = make_3d_packing_prompt(inst, instance_name=base)
    prompt_path = save_prompt_for_instance(prompt, prompts_dir, base, suffix="_3dpack_prompt.txt")

    # Send to LLM
    prompt_for_llm = load_prompt(prompt_path)
    answer_text = send_to_llm(prompt_for_llm, model_selected=model_selected)
    ans_path = save_llm_answer(answer_text, prompts_dir, base, model_selected=model_selected)

    # Parse + validate
    okp, is_yes, asg, tokens = parse_llm_answer_json_packing(answer_text)
    if okp and is_yes:
        ok, reason = verify_llm_assignment_against_cnf(clauses_3, asg)
        is_sat_truth = ok
        if not ok:
            print("[DBG] assignment check failed:", reason)
    else:
        is_sat_truth = False
    return {
        "parsed": okp,
        "llm_yes": is_yes,
        "assignment": asg,
        "tokens": tokens,
        "inst_path": inst_path,
        "prompt_path": prompt_path,
        "answer_path": ans_path,
        "validated_sat": is_sat_truth
    }

# ------------------ Secure send_to_llm: remove hard-coded key ------------------
def send_to_llm(
    prompt_text: str,
    model_selected: str = "gpt-4o",
    system_msg: str = "You are an expert on combinatorial packing.",
    temperature: float = None,
) -> str:
    """
    No hard-coded API key. Use env var OPENAI_API_KEY.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        print("[send_to_llm] OpenAI SDK not installed. pip install openai")
        print("Error:", e)
        return None
    import os, json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[send_to_llm] Missing OPENAI_API_KEY in environment.")
        return None

    client = OpenAI(api_key=api_key)  # default base_url

    name = model_selected.lower()
    looks_restricted = name.endswith('gpt-5')  # adjust if you need model-specific handling

    if looks_restricted:
        user_content = f"[SYSTEM]: {system_msg}\n\n{prompt_text}"
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]

    kwargs = dict(model=model_selected, messages=messages)
    if (temperature is not None) and (not looks_restricted):
        kwargs["temperature"] = float(temperature)

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e)
        if "temperature" in msg and ("unsupported" in msg.lower() or "does not support" in msg.lower()):
            kwargs.pop("temperature", None)
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e2:
                print("[send_to_llm] LLM call failed (retry):", e2)
                return None
        else:
            print("[send_to_llm] LLM call failed:", e)
            return None

    choice0 = resp.choices[0]
    if hasattr(choice0, "message") and hasattr(choice0.message, "content"):
        return choice0.message.content
    return getattr(choice0, "text", None)



# ========= Batch 工具：端点/构体/校验/提交/解析 =========
from tqdm import tqdm
from openai import OpenAI

def endpoint_for(model: str) -> str:
    # 约定：gpt-5 / o1 / o3 用 /v1/responses；其它用 /v1/chat/completions
    name = model.lower()
    return "/v1/responses" if name.startswith(("gpt-5","o1","o3")) else "/v1/chat/completions"

def body_chat_for_packing(model: str, prompt: str, system_msg: str) -> dict:
    constrained = model.lower().startswith(("gpt-5","o1","o3"))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    }
    if constrained:
        body["max_completion_tokens"] = 4096
    else:
        body["max_tokens"] = 4096
        body["temperature"] = 0
    return body

def body_responses_for_packing(model: str, prompt: str, system_msg: str) -> dict:
    # Responses 端点：把规则放 instructions，实例放 input
    body = {
        "model": model,
        "instructions": (
            system_msg + "\n"
            "Return ONLY a one-line JSON: "
            '{"answer":"YES"|"NO","assignment":{"x1":true/false,...},"tokens":[[x,y,z],...],"explain":"..."} '
            "If unsure, return NO with empty assignment/tokens."
        ),
        "input": prompt,
        "max_output_tokens": 4096,
        "text": {"verbosity":"low","format":{"type":"text"}},
        "tool_choice":"none",
        "reasoning":{"effort":"low"},
    }
    return body

def parse_batch_content_text(obj):
    # 兼容 /v1/responses 和 /v1/chat/completions 的文本抽取
    resp = obj.get("response", {}) or {}
    body = resp.get("body", {}) or {}
    ot = body.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot
    if "choices" in body:
        ch = (body.get("choices") or [])
        if ch:
            msg = ch[0].get("message", {}) or {}
            if isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(msg.get("content"), list):
                parts = []
                for p in msg["content"]:
                    if isinstance(p, dict) and p.get("text"): parts.append(p["text"])
                    elif isinstance(p, str): parts.append(p)
                if parts: return "\n".join(parts)
    if "output" in body:
        texts=[]
        for block in body.get("output", []):
            if block.get("type")=="message":
                for c in block.get("content",[]) or []:
                    if isinstance(c,dict) and c.get("text"): texts.append(c["text"])
                    elif isinstance(c,str): texts.append(c)
        if texts: return "\n".join(texts).strip()
    try:
        return obj["response"]["choices"][0]["message"]["content"]
    except Exception:
        return None

def validate_jsonl(jsonl_path, expected_endpoint):
    seen=set(); total=0
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            total+=1
            obj=json.loads(line)
            cid=obj.get("custom_id"); url=obj.get("url"); body=obj.get("body",{}) or {}
            if (not cid) or (cid in seen): raise ValueError(f"JSONL custom_id 问题 at line {ln}")
            seen.add(cid)
            if url!=expected_endpoint: raise ValueError(f"url 不匹配 at line {ln}")
            if expected_endpoint=="/v1/chat/completions":
                if not body.get("model") or not body.get("messages"): raise ValueError(f"body 缺字段 at {ln}")
            else:
                if not body.get("model") or not body.get("input") or not body.get("max_output_tokens"):
                    raise ValueError(f"body 缺字段 at {ln}")
    print(f"[OK] JSONL validated: {jsonl_path}, lines={total}")

def submit_batch_and_wait(client, input_file_id, endpoint, window="24h"):
    batch = client.batches.create(input_file_id=input_file_id, endpoint=endpoint, completion_window=window)
    print("== Batch =="); print("id:", batch.id); print("status:", batch.status)
    while batch.status in ("validating","in_progress","finalizing"):
        print("Batch status:", batch.status, "… waiting 30s")
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print("id:", batch.id, "status:", batch.status)
    return batch



import glob

def answer_path_for(out_root: str, file: str, model_selected: str):
    """
    返回缓存答案文件路径（若存在）。优先匹配你当前的 prompts/answers 命名。
    """
    stem = os.path.splitext(file)[0]
    candidates = [
        os.path.join(out_root, "prompts", "answers", f"{stem}_llm_answer_{model_selected}.txt"),
        os.path.join(out_root, "answers", f"{stem}_llm_answer_{model_selected}.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None

def load_cached_llm_answer_text(ans_path: str) -> str:
    try:
        with open(ans_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


import os
import json
import pandas as pd


def summary_json_to_excel(
        json_path: str,
        excel_path: str = None,
        per_sample_csv_path: str = None
):
    """
    将 evaluate_dir_with_ground_truth_from_cache 生成的 summary JSON
    转为一个多工作表的 Excel 文件。

    工作表包含：
      - Overview：模型名与样本数
      - Accuracy：两套预测（llm_yes / assignment_verified）的总体准确率
      - Confusion：两套预测的混淆矩阵（TP/FP/FN/TN）
      - PerClass：两套预测在 SAT / UNSAT 两个类别的指标（precision/recall/F1 等，如有）
      - PerSample（可选）：逐样本结果 CSV（如果提供 per_sample_csv_path 且文件存在）
    """
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 默认保存到与 JSON 同目录，文件名相同但扩展名为 .xlsx
    if excel_path is None:
        base = os.path.splitext(json_path)[0]
        excel_path = base + ".xlsx"

    # 概览
    overview_df = pd.DataFrame([{
        "model": summary.get("model", ""),
        "num_samples": summary.get("num_samples", 0)
    }])

    # 提取两套指标块
    block_yes = summary.get("pred_llm_yes", {}) or {}
    block_asg = summary.get("pred_assignment_verified", {}) or {}

    def unpack_block(name: str, block: dict):
        """将一个指标块拆成三张小表：Accuracy / Confusion / PerClass"""
        acc = block.get("acc", None)
        confusion = block.get("confusion", {}) or {}
        sat = block.get("SAT", {}) or {}
        unsat = block.get("UNSAT", {}) or {}

        acc_df = pd.DataFrame([{"metric": name, "accuracy": acc}])

        # 混淆矩阵字段名常见为 TP/FP/FN/TN；若你的 binary_metrics 使用不同键名，可在此处映射
        conf_row = {"metric": name}
        # 保障四项都在列里，即使缺失也补 NaN
        for k in ["TP", "FP", "FN", "TN"]:
            conf_row[k] = confusion.get(k)
        confusion_df = pd.DataFrame([conf_row])

        # 分类别指标（把任意已有键全部带上）
        sat_row = {"metric": name, "class": "SAT"}
        sat_row.update(sat)
        unsat_row = {"metric": name, "class": "UNSAT"}
        unsat_row.update(unsat)
        perclass_df = pd.DataFrame([sat_row, unsat_row])

        return acc_df, confusion_df, perclass_df

    acc_yes, conf_yes, perclass_yes = unpack_block("pred_llm_yes", block_yes)
    acc_asg, conf_asg, perclass_asg = unpack_block("pred_assignment_verified", block_asg)

    # 纵向合并，分别构成三张总表
    accuracy_df = pd.concat([acc_yes, acc_asg], ignore_index=True)
    confusion_df = pd.concat([conf_yes, conf_asg], ignore_index=True)
    perclass_df = pd.concat([perclass_yes, perclass_asg], ignore_index=True)

    # 写入 Excel（多工作表）
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        accuracy_df.to_excel(writer, sheet_name="Accuracy", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion", index=False)
        perclass_df.to_excel(writer, sheet_name="PerClass", index=False)

        # 如果给了逐样本 CSV，就一并写入
        if per_sample_csv_path and os.path.exists(per_sample_csv_path):
            try:
                df_samples = pd.read_csv(per_sample_csv_path)
                df_samples.to_excel(writer, sheet_name="PerSample", index=False)
            except Exception as e:
                # 若读取失败，留个提示页
                pd.DataFrame([{"error": str(e)}]).to_excel(
                    writer, sheet_name="PerSample_ERROR", index=False
                )

    print(f"Saved Excel -> {excel_path}")
    return excel_path


# ===== 用法示例 =====
# 假设你的评估函数返回了 (csv_path, json_path, summary)
# csv_path, json_path, summary = evaluate_dir_with_ground_truth_from_cache(...)

# 直接调用（把下面两行改成你的实际路径）
# json_path = r"/path/to/summary_metrics.json"
# csv_path  = r"/path/to/per_sample_results.csv"
# summary_json_to_excel(json_path, per_sample_csv_path=csv_path)


def evaluate_dir_with_ground_truth_from_cache(model_selected: str,
                                              input_dir: str,
                                              out_root: str,
                                              per_sample_csv: str = "per_sample_results.csv",
                                              summary_json: str  = "summary_metrics.json",
                                              skip_missing: bool = True):
    """
    评估阶段不再请求 LLM：
      - 读取 DIMACS 求 ground truth (PySAT)
      - 仅从缓存答案文件中读取 LLM 输出并解析
      - 计算 Pred-A（llm_yes）与 Pred-B（assignment-verified）两套指标
      - 将逐样本结果写 CSV、汇总指标写 JSON
    """
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, per_sample_csv)
    json_path = os.path.join(out_root, summary_json)

    # CSV header
    with open(csv_path, "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow([
            "filename", "model", "N_meta", "alpha_meta",
            "ground_truth", "pred_llm_yes", "pred_assignment_verified", "source_answer_path"
        ])

    y_true, pred_yes, pred_asg = [], [], []
    files = [f for f in sorted(os.listdir(input_dir)) if os.path.isfile(os.path.join(input_dir, f))]

    processed = 0
    for i, file in enumerate(files, 1):
        fp = os.path.join(input_dir, file)

        # 1) ground truth (PySAT)
        clauses_int = read_dimacs(fp)
        truth_sat = cnf_truth_with_pysat(clauses_int)

        # 2) 找缓存答案（不发请求）
        ans_path = answer_path_for(out_root, file, model_selected)
        if not ans_path:
            msg = f"[WARN] cached answer missing -> skip: {file}"
            print(msg)
            if skip_missing:
                continue
            else:
                # 没有答案时可选择给一个保守预测（UNSAT）
                ans_text = None
        else:
            ans_text = load_cached_llm_answer_text(ans_path)

        # 3) 解析缓存答案为 llm_yes / assignment
        okp, is_yes, asg, tokens = parse_llm_answer_json_packing(ans_text or "")
        # Pred-A: llm_yes
        p_yes = bool(is_yes)

        # Pred-B: assignment-verified（规范化键名，并对 3-CNF 验证）
        asg_norm = normalize_assignment_keys_for_eval(asg or {})
        clauses_3 = int_clauses_to_str_3cnf(clauses_int)
        p_asg = bool(p_yes and eval_3cnf_assignment(clauses_3, asg_norm))

        # 4) 写 per-sample CSV
        N_meta, alpha_meta = parse_meta_from_filename(file)
        with open(csv_path, "a", newline="", encoding="utf-8") as wf:
            w = csv.writer(wf)
            w.writerow([
                file,
                model_selected,
                N_meta if N_meta is not None else "",
                alpha_meta if alpha_meta is not None else "",
                "SAT" if truth_sat else "UNSAT",
                "SAT" if p_yes else "UNSAT",
                "SAT" if p_asg else "UNSAT",
                ans_path or "",
            ])

        # 5) 指标累加
        y_true.append(truth_sat)
        pred_yes.append(p_yes)
        pred_asg.append(p_asg)
        processed += 1

        print(f"[{i}/{len(files)}] {file}  truth={truth_sat}  LLM-YES={p_yes}  ASSIGNMENT-OK={p_asg}")

    # 6) 计算汇总指标
    if processed == 0:
        print("[INFO] No samples processed in evaluation (no cached answers found?).")
        summary = {
            "model": model_selected,
            "num_samples": 0,
            "note": "No samples evaluated; cached answers missing."
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return csv_path, json_path, summary

    m_yes = binary_metrics(y_true, pred_yes)
    m_asg = binary_metrics(y_true, pred_asg)

    summary = {
        "model": model_selected,
        "num_samples": len(y_true),
        "pred_llm_yes": {
            "acc": m_yes["acc"],
            "confusion": m_yes["confusion"],
            "SAT":   m_yes["SAT"],
            "UNSAT": m_yes["UNSAT"],
        },
        "pred_assignment_verified": {
            "acc": m_asg["acc"],
            "confusion": m_asg["confusion"],
            "SAT":   m_asg["SAT"],
            "UNSAT": m_asg["UNSAT"],
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Saved (cache-based eval, no API calls) ===")
    print("Per-sample CSV:", csv_path)
    print("Summary JSON  :", json_path)

    summary_json_to_excel(json_path, per_sample_csv_path=csv_path)

    return csv_path, json_path, summary


# ------------------ Driver (reuse your directory conventions) ------------------
def read_cnf_and_run_3d_packing():

    model_list = ['gpt-3.5-turbo']
    model_list = ['gpt-3.5-turbo-0125' ,'chatgpt-4o-latest', 'gpt-4.1'  ]  #  ,'o3-mini' 'o1' , ,'gpt-5' # or anything you are testing
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/cnf_to_3D_packing'

    for model_selected in model_list:
        # for N in [5, 8, 10, 25, 50, 60]:
        for N in [8, 10, 25, 50,]:
            N = str(N)
            dir_name = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
            input_dir  = os.path.join(O1_input_dir_root, dir_name)
            out_root   = os.path.join(output_dir_root, dir_name + f'_3dpack_{model_selected}')
            os.makedirs(out_root, exist_ok=True)

            total = correct_yesno = correct_yes_asg = 0

            for file in sorted(os.listdir(input_dir)):
                fp = os.path.join(input_dir, file)
                if not os.path.isfile(fp):
                    continue

                # --- 新增：若已存在缓存答案，则跳过在线调用 ---
                cached_ans_path = answer_path_for(out_root, file, model_selected)
                if cached_ans_path and os.path.getsize(cached_ans_path) > 0:
                    ans_text = load_cached_llm_answer_text(cached_ans_path)
                    okp, is_yes, asg, tokens = parse_llm_answer_json_packing(ans_text or "")
                    asg = normalize_assignment_keys_for_eval(asg or {})

                    clauses_int = read_dimacs(fp)
                    clauses_3 = int_clauses_to_str_3cnf(clauses_int)
                    if okp and is_yes:
                        ok, reason = verify_llm_assignment_against_cnf(clauses_3, asg)
                        is_sat_truth = ok
                        if not ok:
                            print("[DBG] assignment check failed:", reason)
                    else:
                        is_sat_truth = False
                    total += 1
                    if is_yes == is_sat_truth:
                        correct_yesno += 1
                    if is_sat_truth and is_yes:
                        correct_yes_asg += 1

                    print(f"[SKIP] Use cached: {file}")
                    print(f"      parsed={okp} llm_yes={is_yes} validated_sat={is_sat_truth}")
                    print(f"      cached_answer: {cached_ans_path}")
                    print(f"[ACC] YES/NO so far: {correct_yesno}/{total} = {correct_yesno/total:.3f}")
                    if total:
                        print(f"[ACC] YES-with-correct-asg: {correct_yes_asg}/{total} = {correct_yes_asg/total:.3f}")
                    continue
                # --------------------------------------------------

                # 缓存不存在，才真正调用 LLM
                res = process_one_file_3d_packing(fp,
                                                  packing_out_dir=out_root,
                                                  model_selected=model_selected)

                if res["parsed"]:
                    total += 1
                    if res["llm_yes"] == res["validated_sat"]:
                        correct_yesno += 1
                    if res["validated_sat"] and res["llm_yes"]:
                        correct_yes_asg += 1

                    print(f"[{file}] parsed={res['parsed']} llm_yes={res['llm_yes']} validated_sat={res['validated_sat']}")
                    print(f"  inst:   {res['inst_path']}")
                    print(f"  prompt: {res['prompt_path']}")
                    print(f"  answer: {res['answer_path']}")
                    print(f"[ACC] YES/NO so far: {correct_yesno}/{total} = {correct_yesno/total:.3f}")
                    if total:
                        print(f"[ACC] YES-with-correct-asg: {correct_yes_asg}/{total} = {correct_yes_asg/total:.3f}")

            if total:
                print(f"\n[FINAL] Model={model_selected}, N={N}")
                print(f"  YES/NO accuracy: {correct_yesno}/{total} = {correct_yesno/total:.3f}")
                print(f"  YES-with-correct-assignment: {correct_yes_asg}/{total} = {correct_yes_asg/total:.3f}")

            # 仅用缓存做整体评估（不会触发任何 API 调用）
            evaluate_dir_with_ground_truth_from_cache(
                model_selected=model_selected,
                input_dir=input_dir,
                out_root=out_root,
                per_sample_csv=f"{model_selected}_N{N}_per_sample_results.csv",
                summary_json=f"{model_selected}_N{N}_summary_metrics.json",
                skip_missing=True,
            )



def build_packing_batch_jsonl(model_selected: str,
                              input_dir: str,
                              out_root: str,
                              jsonl_path: str,
                              system_msg: str = "You are an expert in combinatorial packing and constraint solving."):
    """
    遍历 input_dir 下的 DIMACS，生成：
      - instances/*.json
      - prompts/*.txt
      - batch jsonl（每个样本一行请求）
    返回 (task_meta dict, endpoint)
    """
    os.makedirs(out_root, exist_ok=True)
    instances_dir = os.path.join(out_root, "instances")
    prompts_dir   = os.path.join(out_root, "prompts")
    answers_dir   = os.path.join(out_root, "answers")
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(prompts_dir,   exist_ok=True)
    os.makedirs(answers_dir,   exist_ok=True)

    endpoint = endpoint_for(model_selected)
    task_meta = {}
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as wf:
        for file in sorted(os.listdir(input_dir)):
            fp = os.path.join(input_dir, file)
            if not os.path.isfile(fp):
                continue

            # 1) 读 DIMACS → 3CNF → 3D packing
            clauses_int = read_dimacs(fp)
            if not clauses_int:
                continue
            clauses_3 = int_clauses_to_str_3cnf(clauses_int)
            inst = build_3d_packing_from_3cnf(clauses_3)

            # 2) 保存 instance / prompt
            inst_path = save_3d_packing_instance(inst, instances_dir, file)
            prompt = make_3d_packing_prompt(inst, instance_name=file)
            prompt_path = save_prompt_for_instance(prompt, prompts_dir, file, suffix="_3dpack_prompt.txt")

            # 3) 为本样本写一条 batch 请求
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()

            if endpoint == "/v1/responses":
                body = body_responses_for_packing(model_selected, prompt_text, system_msg)
            else:
                body = body_chat_for_packing(model_selected, prompt_text, system_msg)

            custom_id = f"pack_{model_selected}_{os.path.splitext(file)[0]}"
            task_meta[custom_id] = {
                "file": file,
                "inst_path": inst_path,
                "prompt_path": prompt_path,
                "answers_dir": answers_dir,
                "clauses_3": clauses_3,
                "out_txt": os.path.join(answers_dir, os.path.splitext(file)[0] + f"_llm_answer_{model_selected}.txt"),
            }

            row = {"custom_id": custom_id, "method":"POST", "url": endpoint, "body": body}
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[BUILD] {jsonl_path} built, tasks={written}, endpoint={endpoint}")
    return task_meta, endpoint


def run_packing_batch_and_collect(model_selected: str, jsonl_path: str, task_meta: dict, endpoint: str, api_key: str=None):
    client = OpenAI(api_key=(api_key or os.getenv("OPENAI_API_KEY")))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")

    # 1) 校验并上传
    validate_jsonl(jsonl_path, endpoint)
    upload = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch  = submit_batch_and_wait(client, upload.id, endpoint, window="24h")

    if getattr(batch, "status", None) != "completed":
        raise RuntimeError(f"Batch 未完成，status={batch.status}")
    out_id = getattr(batch, "output_file_id", None)
    if not out_id:
        raise RuntimeError("Batch 完成但没有 output_file_id")

    # 2) 读取 batch 输出，逐条解析与写回
    out_text = client.files.content(out_id).text
    total=ok_parse=ok_sat=0
    for line in out_text.splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        cid = obj.get("custom_id","")
        meta = task_meta.get(cid)
        if not meta: continue

        text = parse_batch_content_text(obj) or ""
        # 保存原始答复
        with open(meta["out_txt"], "w", encoding="utf-8") as f:
            f.write(text.strip())

        # 解析一行 JSON（我们要求 LLM 输出 one-line JSON）
        okp, is_yes, asg, tokens = parse_llm_answer_json_packing(text)
        total += 1
        if okp:
            ok_parse += 1
            # 用 CNF 真值校验（YES 才有 assignment）
            is_sat_truth = eval_3cnf_assignment(meta["clauses_3"], asg) if is_yes else False
            if is_yes == is_sat_truth:
                ok_sat += 1

    print(f"[RESULT] total={total}, parsed_ok={ok_parse}, yes/no correct vs CNF={ok_sat}")
    return total, ok_parse, ok_sat


def read_cnf_and_run_3d_packing_batch():
    # 用你想测的模型；注意 gpt-3.5-turbo 已下线，建议 chatgpt-4o-latest / gpt-4o / gpt-4.1 / gpt-5
    model_list = ['gpt-3.5-turbo']
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root   = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/cnf_to_3D_packing'

    for model_selected in model_list:
        for N in [8, 10, 25, 50]:   # 你原来就跑 N=5；要多 N 就加上
            N = str(N)
            dir_name  = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
            input_dir = os.path.join(O1_input_dir_root, dir_name)
            out_root  = os.path.join(output_dir_root, dir_name + f'_3dpack_{model_selected}')
            os.makedirs(out_root, exist_ok=True)

            # A) 构建 batch jsonl（同时生成 instances / prompts）
            jsonl_path = os.path.join(out_root, f"batch_tasks_{model_selected}.jsonl")
            task_meta, endpoint = build_packing_batch_jsonl(
                model_selected=model_selected,
                input_dir=input_dir,
                out_root=out_root,
                jsonl_path=jsonl_path,
                system_msg="You are an expert in combinatorial packing."
            )

            if not task_meta:
                print(f"[{model_selected}] 没有可提交的任务。")
                continue

            # B) 提交 batch 并收集落盘到 answers/
            run_packing_batch_and_collect(
                model_selected=model_selected,
                jsonl_path=jsonl_path,
                task_meta=task_meta,
                endpoint=endpoint,
                api_key=None  # 用环境变量
            )


import re, csv, json, os
from typing import List, Dict, Any

# ====== Ground truth via PySAT ======
def cnf_truth_with_pysat(clauses_int) -> bool:
    """True = SAT, False = UNSAT"""
    try:
        from pysat.formula import CNF
        from pysat.solvers import Minisat22
    except Exception as e:
        raise RuntimeError("Please install PySAT: pip install python-sat[pblib,aiger]") from e
    formula = CNF(from_clauses=clauses_int)
    with Minisat22(bootstrap_with=formula.clauses) as s:
        return s.solve()

# ====== parse N / alpha from filename ======
def parse_meta_from_filename(filename: str):
    """
    Try to extract N and alpha; return (N:int|None, alpha:float|None).
    Matches e.g.:  ..._N5_..._alpha3.5_...
    """
    N = None
    alpha = None
    m = re.search(r'[\W_]N(\d+)\b', filename, flags=re.IGNORECASE)
    if m:
        try:
            N = int(m.group(1))
        except Exception:
            N = None
    m = re.search(r'alpha([0-9]+(?:\.[0-9]+)?)', filename, flags=re.IGNORECASE)
    if m:
        try:
            alpha = float(m.group(1))
        except Exception:
            alpha = None
    return N, alpha

# ====== safe division ======
def _safe_div(a, b):
    return (a / b) if b else 0.0

# ====== compute metrics ======
def binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """
    True=SAT, False=UNSAT.
    Return ACC and per-class (SAT/UNSAT) precision/recall/F1 + confusion.
    """
    assert len(y_true) == len(y_pred)
    Ntot = len(y_true)
    TP = sum(1 for t, p in zip(y_true, y_pred) if t and p)          # SAT predicted SAT
    TN = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))  # UNSAT predicted UNSAT
    FP = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)    # UNSAT predicted SAT
    FN = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))    # SAT predicted UNSAT

    acc = _safe_div(TP + TN, Ntot)

    # SAT as positive
    prec_sat = _safe_div(TP, (TP + FP))
    rec_sat  = _safe_div(TP, (TP + FN))
    f1_sat   = _safe_div(2 * prec_sat * rec_sat, (prec_sat + rec_sat)) if (prec_sat + rec_sat) else 0.0

    # UNSAT as positive (swap roles)
    prec_unsat = _safe_div(TN, (TN + FN))
    rec_unsat  = _safe_div(TN, (TN + FP))
    f1_unsat   = _safe_div(2 * prec_unsat * rec_unsat, (prec_unsat + rec_unsat)) if (prec_unsat + rec_unsat) else 0.0

    return {
        "N": Ntot,
        "acc": acc,
        "confusion": {"TP_SAT": TP, "TN_UNSAT": TN, "FP_on_UNSAT": FP, "FN_on_SAT": FN},
        "SAT":   {"precision": prec_sat,   "recall": rec_sat,   "f1": f1_sat},
        "UNSAT": {"precision": prec_unsat, "recall": rec_unsat, "f1": f1_unsat},
    }

# ====== normalize assignment keys like {"1":true} -> {"x1":true} ======
def normalize_assignment_keys_for_eval(asg: Dict[str, bool]) -> Dict[str, bool]:
    out = {}
    for k, v in (asg or {}).items():
        ks = str(k).strip()
        if ks.startswith('x'):
            out[ks] = bool(v)
        elif ks.isdigit():
            out['x' + ks] = bool(v)
        else:
            # keep as-is (fallback)
            out[ks] = bool(v)
    return out








if __name__ == "__main__":
    # read_cnf()          # old VC pipeline
    read_cnf_and_run_3d_packing()   # new 3D packing + LLM pipeline
    # read_cnf_and_run_3d_packing_batch()