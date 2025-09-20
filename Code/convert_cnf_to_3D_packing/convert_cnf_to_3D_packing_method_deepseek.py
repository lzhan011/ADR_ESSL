# --- add project root to sys.path ---
import sys, os, time, re, csv, json, requests
from typing import List, Dict, Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # -> Satisfiability_Solvers
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 你自己的工具/转换函数（保持不变）
from Code.convert_cnf_to_vertex_cover.convert_cnf_to_vertex_cover_method_2 import  *
import pandas as pd

# ========================= DeepSeek API 基础封装 =========================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_URL = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

def _deepseek_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    key = os.environ["DEEPSEEK_API_KEY"]
    if not key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment.")
    return key

def _map_model_to_deepseek(name: str) -> str:
    """
    兼容原来传入的各种模型名；统一映射到 deepseek 的两个主要模型。
    """
    n = (name or "").lower().strip()
    alias = {
        "gpt-3.5-turbo": "deepseek-chat",
        "gpt-3.5-turbo-0125": "deepseek-chat",
        "chatgpt-4o-latest": "deepseek-chat",
        "gpt-4o": "deepseek-chat",
        "gpt-4.1": "deepseek-chat",
        "gpt-4-turbo": "deepseek-chat",
        "o1": "deepseek-reasoner",
        "o3-mini": "deepseek-chat",
        "gpt-5": "deepseek-reasoner",
        # 直接使用 deepseek 名称时原样返回
        "deepseek-chat": "deepseek-chat",
        "deepseek-reasoner": "deepseek-reasoner",
    }
    return alias.get(n, "deepseek-reasoner")

def deepseek_chat_completion(messages: List[Dict[str, str]], model: str, temperature: float = None) -> str:
    """
    直接调用 DeepSeek 的 /v1/chat/completions。
    返回字符串（优先 message.content）。
    """
    headers = {
        "Authorization": f"Bearer {_deepseek_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _map_model_to_deepseek(model),
        "messages": messages,
        "max_tokens": 4096,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)

    resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    # 兼容 OpenAI 风格
    try:
        ch0 = data["choices"][0]
        msg = ch0.get("message", {})
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
        # 兜底
        return ch0.get("text", "")
    except Exception:
        return str(data)

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
                "Packing feasible <=> the 3-CNF is satisfiable."
            ]
        }
    }
    return instance

# ------------------ DIMACS reader (robust) ------------------
def read_dimacs(filepath):
    """
    Parse standard DIMACS CNF:
      c ...
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
            if cur and parts[-1] != '0':
                pass
    return clauses

# ------------------ 3-CNF -> 3D Packing instance (备用) ------------------
def build_3d_packing_from_3cnf_bak(clauses_3):
    vars_set = sorted({literal_var(l) for C in clauses_3 for l in C},
                      key=lambda s: int(s[1:]))
    idx = {v: i+1 for i, v in enumerate(vars_set)}  # 1-based
    n = len(vars_set)
    m = len(clauses_3)

    clause_slots = []
    for j, C in enumerate(clauses_3):
        slots = []
        for lit in C:
            v = literal_var(lit)
            i = idx[v]
            y = 0 if not literal_is_neg(lit) else 1
            slots.append([i, y, j])
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
def make_3d_packing_prompt(instance, instance_name=None):
    X, Y, Z = instance["container"]["size"]
    header = (
        "You are an expert in discrete 3D packing and constraint solving.\n"
        f"- Container (grid): X = {X}, Y = {Y}, Z = {Z}; rotations are NOT allowed and orientations are locked.\n"
        "- Objects come in TWO types (no other types exist):\n"
        "  * Type-A (long rods): for each x-index in {1..X} there are EXACTLY TWO rods, anchored at y=0 and y=1, "
        "each with size [1,1,Z] and spanning cells (x=const, y in {0,1}, z=0..Z-1). "
        "You must SELECT EXACTLY ONE rod per x-index (choose y=0 OR y=1), never both.\n"
        '  * Type-B (unit cubes, "tokens"): there is EXACTLY ONE Type-B object for each depth z=j (j=0..Z-1). '
        "Each Type-B object lists its allowed placements as coordinates in its `allowed_slots` field.\n"
        "- Valid placement rule:\n"
        "  * A Type-B placement at (x,y,z=j) is VALID ONLY if that cell is covered by a SELECTED Type-A rod at the same (x,y).\n"
        "  * You MUST choose coordinates ONLY from the `allowed_slots` provided; do not invent coordinates.\n"
        "- Feasibility means EVERY Type-B object can be placed validly under these rules.\n\n"
        "Return ONLY a ONE-LINE JSON object:\n"
        '  {"answer":"YES"|"NO",'
        ' "assignment":{"1":true/false,"2":true/false,...,"' + str(X) + '":true/false},'
        ' "tokens":[[x,y,z],...],'
        ' "explain":"<= 2 sentences"}\n'
        "Where:\n"
        '  - `assignment` uses STRING KEYS for ALL x-indices 1..X; true => select the rod at y=0, false => select the rod at y=1.\n'
        "  - `tokens` must contain EXACTLY one coordinate for EACH depth layer z=0..Z-1, in ascending z order, "
        "and each coordinate must be one of that layer's `allowed_slots`.\n"
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
    asg = normalize_assignment_keys_for_eval(asg_raw or {})
    vars_in_formula = { literal_var(l) for C in clauses_3 for l in C }
    missing = [v for v in vars_in_formula if v not in asg]
    if missing:
        return False, f"assignment missing vars: {missing[:10]}{'...' if len(missing)>10 else ''}"
    ok = eval_3cnf_assignment(clauses_3, asg)
    return (ok, "ok" if ok else "violates at least one clause")

# ------------------ End-to-end: per-file 3D packing prompt/run ------------------
def process_one_file_3d_packing(filepath, packing_out_dir=None, model_selected=""):
    clauses_int = read_dimacs(filepath)
    clauses_3 = int_clauses_to_str_3cnf(clauses_int)  # ensures 3-CNF

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

    # Send to LLM (DeepSeek)
    system_msg = "You are an expert on combinatorial packing."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    answer_text = deepseek_chat_completion(messages, model_selected, temperature=0)
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

# ------------------ Secure send_to_llm: DeepSeek 版本 ------------------
def send_to_llm(
    prompt_text: str,
    model_selected: str = "gpt-4o",   # 兼容旧默认；内部会映射到 deepseek
    system_msg: str = "You are an expert on combinatorial packing.",
    temperature: float = None,
) -> str:
    """
    DeepSeek-only 调用；使用 env var DEEPSEEK_API_KEY。
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    return deepseek_chat_completion(messages, model_selected, temperature=temperature)

# ========= Batch 工具（本实现为“本地顺序执行版”，不使用 OpenAI batch） =========

def endpoint_for(model: str) -> str:
    # DeepSeek 统一用 chat/completions
    return "/v1/chat/completions"

def body_chat_for_packing(model: str, prompt: str, system_msg: str) -> dict:
    return {
        "model": _map_model_to_deepseek(model),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }

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
            if not body.get("model") or not body.get("messages"):
                raise ValueError(f"body 缺字段 at {ln}")
    print(f"[OK] JSONL validated: {jsonl_path}, lines={total}")

def submit_batch_and_wait(*args, **kwargs):
    raise NotImplementedError("This script uses a local sequential runner for DeepSeek (no remote batch API).")

def answer_path_for(out_root: str, file: str, model_selected: str):
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

def summary_json_to_excel(
        json_path: str,
        excel_path: str = None,
        per_sample_csv_path: str = None
):
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if excel_path is None:
        base = os.path.splitext(json_path)[0]
        excel_path = base + ".xlsx"
    overview_df = pd.DataFrame([{
        "model": summary.get("model", ""),
        "num_samples": summary.get("num_samples", 0)
    }])
    block_yes = summary.get("pred_llm_yes", {}) or {}
    block_asg = summary.get("pred_assignment_verified", {}) or {}

    def unpack_block(name: str, block: dict):
        acc = block.get("acc", None)
        confusion = block.get("confusion", {}) or {}
        sat = block.get("SAT", {}) or {}
        unsat = block.get("UNSAT", {}) or {}
        acc_df = pd.DataFrame([{"metric": name, "accuracy": acc}])
        conf_row = {"metric": name}
        for k in ["TP", "FP", "FN", "TN", "TP_SAT", "TN_UNSAT", "FP_on_UNSAT", "FN_on_SAT"]:
            if k in confusion:
                conf_row[k] = confusion.get(k)
        confusion_df = pd.DataFrame([conf_row])
        sat_row = {"metric": name, "class": "SAT"};  sat_row.update(sat)
        unsat_row = {"metric": name, "class": "UNSAT"}; unsat_row.update(unsat)
        perclass_df = pd.DataFrame([sat_row, unsat_row])
        return acc_df, confusion_df, perclass_df

    acc_yes, conf_yes, perclass_yes = unpack_block("pred_llm_yes", block_yes)
    acc_asg, conf_asg, perclass_asg = unpack_block("pred_assignment_verified", block_asg)

    accuracy_df = pd.concat([acc_yes, acc_asg], ignore_index=True)
    confusion_df = pd.concat([conf_yes, conf_asg], ignore_index=True)
    perclass_df = pd.concat([perclass_yes, perclass_asg], ignore_index=True)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        accuracy_df.to_excel(writer, sheet_name="Accuracy", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion", index=False)
        perclass_df.to_excel(writer, sheet_name="PerClass", index=False)
        if per_sample_csv_path and os.path.exists(per_sample_csv_path):
            try:
                df_samples = pd.read_csv(per_sample_csv_path)
                df_samples.to_excel(writer, sheet_name="PerSample", index=False)
            except Exception as e:
                pd.DataFrame([{"error": str(e)}]).to_excel(
                    writer, sheet_name="PerSample_ERROR", index=False
                )
    print(f"Saved Excel -> {excel_path}")
    return excel_path

def evaluate_dir_with_ground_truth_from_cache(model_selected: str,
                                              input_dir: str,
                                              out_root: str,
                                              per_sample_csv: str = "per_sample_results.csv",
                                              summary_json: str  = "summary_metrics.json",
                                              skip_missing: bool = True):
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, per_sample_csv)
    json_path = os.path.join(out_root, summary_json)

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
        clauses_int = read_dimacs(fp)
        truth_sat = cnf_truth_with_pysat(clauses_int)

        ans_path = answer_path_for(out_root, file, model_selected)
        if not ans_path:
            msg = f"[WARN] cached answer missing -> skip: {file}"
            print(msg)
            if skip_missing:
                continue
            else:
                ans_text = None
        else:
            ans_text = load_cached_llm_answer_text(ans_path)

        okp, is_yes, asg, tokens = parse_llm_answer_json_packing(ans_text or "")
        p_yes = bool(is_yes)

        asg_norm = normalize_assignment_keys_for_eval(asg or {})
        clauses_3 = int_clauses_to_str_3cnf(clauses_int)
        p_asg = bool(p_yes and eval_3cnf_assignment(clauses_3, asg_norm))

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

        y_true.append(truth_sat)
        pred_yes.append(p_yes)
        pred_asg.append(p_asg)
        processed += 1

        print(f"[{i}/{len(files)}] {file}  truth={truth_sat}  LLM-YES={p_yes}  ASSIGNMENT-OK={p_asg}")

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


def get_pairs(input_dir):
    # --- 先按“成对”分组：key = 去掉后缀 _RC2_fixed 的文件基名 ---
    all_files = [
        f for f in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".cnf")
    ]

    pairs = {}  # key -> {"orig": filename(无 _RC2_fixed), "fixed": filename(有 _RC2_fixed)}
    for f in all_files:
        stem = os.path.splitext(f)[0]
        if stem.endswith("_RC2_fixed"):
            key = stem[: -len("_RC2_fixed")]
            pairs.setdefault(key, {}).update({"fixed": f})
        else:
            key = stem
            pairs.setdefault(key, {}).update({"orig": f})

    return pairs



def _is_nonempty_file(path: str, require_non_whitespace: bool = True) -> bool:
    """True 当且仅当文件为非空文件；若 require_non_whitespace=True，还要求存在非空白字符。"""
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) <= 0:
            return False
        if not require_non_whitespace:
            return True
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    return True
        return False
    except Exception:
        return False


def get_pairs(input_dir: str, require_non_whitespace: bool = True) -> dict:
    """
    返回形如：
      {
        'cnf_k3_N8_L28_alpha3.5_inst113.txt': {
            'orig':  'cnf_k3_N8_L28_alpha3.5_inst113_llm_answer_gpt-5.txt',
            'fixed': 'cnf_k3_N8_L28_alpha3.5_inst113_RC2_fixed_llm_answer_gpt-5.txt'
        },
        ...
      }

    规则：
      - key：去掉 “_llm_answer” 及其后的模型名，但 **保留后缀**；同时去掉末尾的 “_RC2_fixed”（若有），用于将一对配在同一 key 下。
      - value：保持原始文件名不变。
      - 若去模型段后基名以 “_RC2_fixed” 结尾 => 归为 fixed；否则归为 orig。
      - 跳过空文件（大小=0 或仅空白）。
    """

    pairs = {}

    # 目录不存在或不是目录：返回空 dict
    if not input_dir or not os.path.isdir(input_dir):
        return pairs

    # 列目录，若出错或为空：返回空 dict
    try:
        entries = os.listdir(input_dir)
    except Exception:
        return pairs
    if not entries:
        return pairs

    # 允许的扩展名（答案通常为 .txt；按需可加入 .cnf）
    exts = {".txt", ".cnf"}
    all_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in exts
    )

    # 用于剥离 “_llm_answer...模型名” 段，同时保留扩展名
    # 例：abc_RC2_fixed_llm_answer_gpt-5.txt -> base_no_model=abc_RC2_fixed, ext=.txt
    pat = re.compile(r"^(?P<base>.+?)(?:_llm_answer.*)?(?P<ext>\.[^.]+)$", re.IGNORECASE)

    pairs = {}
    for f in all_files:
        path = os.path.join(input_dir, f)
        if not _is_nonempty_file(path, require_non_whitespace=require_non_whitespace):
            continue

        m = pat.match(f)
        if not m:
            # 不匹配时跳过（非常规命名）
            continue

        base_no_model = m.group("base")  # 去掉 _llm_answer 及模型段
        ext = m.group("ext").lower()     # 保留后缀

        # 判断是否为 fixed：去模型段后的基名若以 _RC2_fixed 结尾即 fixed
        is_fixed = base_no_model.endswith("_RC2_fixed")

        # 规范化 key：去掉末尾的 _RC2_fixed（若有），并**保留后缀**
        key_base = base_no_model[:-len("_RC2_fixed")] if is_fixed else base_no_model
        key = key_base + ext  # e.g., cnf_k3_N8_..._inst113.txt

        # 归档
        d = pairs.setdefault(key, {})
        if is_fixed:
            d["fixed"] = f
        else:
            d["orig"] = f

    return pairs



def split_pairs_by_completeness(pairs_output: dict):
    """
    将 pairs_output 拆分为：
      - complete_pairs: orig 与 fixed 均存在
      - incomplete_pairs: 仅存在其一（orig XOR fixed）
      - only_orig_pairs: 仅 orig 存在
      - only_fixed_pairs: 仅 fixed 存在
    返回 (complete_pairs, incomplete_pairs, only_orig_pairs, only_fixed_pairs)
    """
    def present(x) -> bool:
        # 既考虑 None 也考虑空串/空白串
        return x is not None and str(x).strip() != ""

    complete_pairs = {}
    incomplete_pairs = {}
    only_orig_pairs = {}
    only_fixed_pairs = {}

    for key, val in (pairs_output or {}).items():
        orig_ok  = present(val.get("orig"))
        fixed_ok = present(val.get("fixed"))

        if orig_ok and fixed_ok:
            complete_pairs[key] = {"orig": val.get("orig"), "fixed": val.get("fixed")}
        elif orig_ok ^ fixed_ok:
            incomplete_pairs[key] = {"orig": val.get("orig"), "fixed": val.get("fixed")}
            if orig_ok:
                only_orig_pairs[key] = {"orig": val.get("orig"), "fixed": val.get("fixed")}
            else:
                only_fixed_pairs[key] = {"orig": val.get("orig"), "fixed": val.get("fixed")}
        # 两个都不存在的 key 直接忽略

    return complete_pairs, incomplete_pairs, only_orig_pairs, only_fixed_pairs



def plan_keys_after_reconcile(pairs_input: dict,
                              complete_pairs: dict,
                              incomplete_pairs: dict,
                              sort_keys: bool = True) -> list:
    """
    - 从 pairs_input 中剔除 complete_pairs 的 key
    - 将剩余的 key 与 incomplete_pairs 的 key 合并为一个列表
    - 保证 incomplete_pairs 的 key 在列表前面（且不重复）
    - 可选按字典序排序

    返回：按优先级排列好的 key 列表
    """
    # 1) 先剔除已完成的 key
    done = set(complete_pairs.keys())
    remaining = [k for k in pairs_input.keys() if k not in done]

    # 2) 把未完成的 key 放前面
    inc_keys = list(incomplete_pairs.keys())

    # 3) 去重合并：优先 incomplete，其次 remaining 中不在 incomplete 的
    inc_set = set(inc_keys)
    tail_keys = [k for k in remaining if k not in inc_set]

    if sort_keys:
        inc_keys = sorted(inc_keys)
        tail_keys = sorted(tail_keys)

    plan = inc_keys + tail_keys
    return plan


# ------------------ Driver：单例/缓存优先 ------------------
def read_cnf_and_run_3d_packing():
    model_list = ['gpt-3.5-turbo-0125' ,'chatgpt-4o-latest', 'gpt-4.1']  # 会映射到 deepseek-chat
    model_list = [ 'deepseek-chat',  ] # 'deepseek-reasoner'
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/cnf_to_3D_packing'

    for model_selected in model_list:
        for N in [5, 8, 10, 25, 50]:
            N = str(N)
            dir_name = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
            input_dir  = os.path.join(O1_input_dir_root, dir_name)
            out_root   = os.path.join(output_dir_root, dir_name + f'_3dpack_{model_selected}')
            os.makedirs(out_root, exist_ok=True)

            total = correct_yesno = correct_yes_asg = 0
            pairs_input = get_pairs(input_dir)
            pairs_output = get_pairs(os.path.join(out_root, 'prompts/answers'))
            complete_pairs, incomplete_pairs, only_orig_pairs, only_fixed_pairs = split_pairs_by_completeness(
                pairs_output)
            # --- 按键有序遍历；每对内部先处理原始，再处理 _RC2_fixed ---

            plan_list = plan_keys_after_reconcile(pairs_input, complete_pairs, incomplete_pairs)
            max_instances_number = 36 - len(list(complete_pairs.keys()))
            write_pair_each_N = 0
            for key in sorted(plan_list):
                write_pair_each_N += 1
                if write_pair_each_N >= max_instances_number:
                    break
                for role in ("orig", "fixed"):
                    file = pairs_input[key].get(role)
                    if not file:
                        continue
                    fp = os.path.join(input_dir, file)
                    if not os.path.isfile(fp):
                        continue

                    # 缓存优先
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

                # 无缓存 -> 实时调用 DeepSeek
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

# ------------------ 构建“本地批量执行”的 JSONL，并顺序执行 ------------------
def build_packing_batch_jsonl(model_selected: str,
                              input_dir: str,
                              out_root: str,
                              jsonl_path: str,
                              system_msg: str = "You are an expert in combinatorial packing and constraint solving."):
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

            clauses_int = read_dimacs(fp)
            if not clauses_int:
                continue
            clauses_3 = int_clauses_to_str_3cnf(clauses_int)
            inst = build_3d_packing_from_3cnf(clauses_3)

            inst_path = save_3d_packing_instance(inst, instances_dir, file)
            prompt = make_3d_packing_prompt(inst, instance_name=file)
            prompt_path = save_prompt_for_instance(prompt, prompts_dir, file, suffix="_3dpack_prompt.txt")

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()

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
    """
    本地顺序执行“批处理”：逐行读取 JSONL，调用 DeepSeek chat/completions，结果落盘。
    """
    validate_jsonl(jsonl_path, endpoint)

    headers = {
        "Authorization": f"Bearer {api_key or _deepseek_api_key()}",
        "Content-Type": "application/json",
    }

    total=ok_parse=ok_sat=0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id","")
            meta = task_meta.get(cid)
            if not meta:
                continue
            body = obj.get("body", {}) or {}

            # 实时调用 DeepSeek
            resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=body, timeout=120)
            if resp.status_code != 200:
                text = f"[ERROR] {resp.status_code}: {resp.text[:400]}"
            else:
                data = resp.json()
                try:
                    ch0 = data["choices"][0]
                    msg = ch0.get("message", {})
                    text = msg.get("content") or ch0.get("text", "")
                except Exception:
                    text = str(data)

            # 保存原始答复
            with open(meta["out_txt"], "w", encoding="utf-8") as wf:
                wf.write((text or "").strip())

            okp, is_yes, asg, tokens = parse_llm_answer_json_packing(text or "")
            total += 1
            if okp:
                ok_parse += 1
                is_sat_truth = eval_3cnf_assignment(meta["clauses_3"], asg) if is_yes else False
                if is_yes == is_sat_truth:
                    ok_sat += 1

    print(f"[RESULT] total={total}, parsed_ok={ok_parse}, yes/no correct vs CNF={ok_sat}")
    return total, ok_parse, ok_sat

def read_cnf_and_run_3d_packing_batch():
    model_list = ['gpt-3.5-turbo']  # 会映射到 deepseek-chat
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root   = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/cnf_to_3D_packing'

    for model_selected in model_list:
        for N in [5, 8, 10, 25, 50]:
            N = str(N)
            dir_name  = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
            input_dir = os.path.join(O1_input_dir_root, dir_name)
            out_root  = os.path.join(output_dir_root, dir_name + f'_3dpack_{model_selected}')
            os.makedirs(out_root, exist_ok=True)

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

            run_packing_batch_and_collect(
                model_selected=model_selected,
                jsonl_path=jsonl_path,
                task_meta=task_meta,
                endpoint=endpoint,
                api_key=None  # 使用环境变量
            )

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
    assert len(y_true) == len(y_pred)
    Ntot = len(y_true)
    TP = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    TN = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    FP = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))

    acc = _safe_div(TP + TN, Ntot)

    prec_sat = _safe_div(TP, (TP + FP))
    rec_sat  = _safe_div(TP, (TP + FN))
    f1_sat   = _safe_div(2 * prec_sat * rec_sat, (prec_sat + rec_sat)) if (prec_sat + rec_sat) else 0.0

    prec_unsat = _safe_div(TN, (TN + FN))
    rec_unsat  = _safe_div(TN, (TN + FP))
    f1_unsat   = _safe_div(2 * prec_unsat * rec_unsat, (prec_unsat + rec_unsat)) if (prec_unsat + rec_unsat) else 0.0

    return {
        "N": Ntot,
        "acc": acc,
        "confusion": {
            "TP_SAT": TP, "TN_UNSAT": TN,
            "FP_on_UNSAT": FP, "FN_on_SAT": FN
        },
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
            out[ks] = bool(v)
    return out

# ============================== MAIN ==============================
if __name__ == "__main__":
    # 默认运行单次顺序流程；如需本地“批处理”，改为调用 read_cnf_and_run_3d_packing_batch()
    read_cnf_and_run_3d_packing()
    # read_cnf_and_run_3d_packing_batch()
