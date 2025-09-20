import os
import re
import math
import numpy as np
import pandas as pd
from pysat.solvers import Minisat22

# ================== 目标目录与输出 ==================
FOLDER1_ROOT = "/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/draw_o1_phase_transition_figures"
FOLDER1_PREFIX = "draw_o1_cnf_alpha_3_6_N_75"

FOLDER2_ROOT = "/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai"
FOLDER2_PREFIX = "cnf_results_openai_o1"

ANALYSIS_DIR = os.path.join(FOLDER2_ROOT, "analysis_o1_assignment_check")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(ANALYSIS_DIR, "assignment_check_summary.csv")
DETAILS_CSV = os.path.join(ANALYSIS_DIR, "assignment_check_details.csv")


# ================== 实用函数 ==================
def _safe_div(num, den):
    return num / den if den else float("nan")


def _mcc_from_counts(tp, fp, fn, tn):
    # 这里不需要 MCC，保留以备扩展
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return float("nan")
    return (tp * tn - fp * fn) / math.sqrt(denom)


# ================== 解析 CNF + 赋值 ==================
def parse_cnf_and_model(filepath, N_hint=None):
    """
    解析单个文件内：
      - DIMACS CNF 子句（从 'p cnf' 后开始）
      - 预测是否 SAT/UNSAT（搜索关键字）
      - 若 SAT，解析 x<i>=<val> 的赋值（支持 1/0/t/f/true/false）
    返回: (clauses: List[List[int]], model: List[int], sat: str|None, nvars: int|None)
    """
    clauses = []
    model_dict = {}
    sat = None
    nvars_hdr = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- 抓 p cnf 行并读取 nvars,nclauses
    p_re = re.compile(r"^\s*p\s+cnf\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE)
    read_clause = False
    for line in lines:
        m = p_re.match(line)
        if m:
            nvars_hdr = int(m.group(1))
            read_clause = True
            continue

        if read_clause:
            # DIMACS 子句段落：只要行里有数字就解析（跳过注释/空行）
            if not any(ch.isdigit() for ch in line):
                # 遇到非数字行，通常意味着子句部分结束
                # 但部分日志格式可能在后面还有数字；此处保持简单策略
                read_clause = False
                continue
            # 将行中数字取出，去掉末尾 0
            toks = line.strip().split()
            one = []
            for t in toks:
                if t == "0":
                    break
                if re.fullmatch(r"-?\d+", t):
                    one.append(int(t))
            if one:
                clauses.append(one)

    # --- 解析 SAT/UNSAT 关键词（大小写无关）
    sat_pat = re.compile(r"\b(SATISFIABLE|UNSATISFIABLE)\b", re.IGNORECASE)
    for line in lines:
        mm = sat_pat.search(line)
        if mm:
            kw = mm.group(1).upper()
            sat = "SATISFIABLE" if kw == "SATISFIABLE" else "UNSATISFIABLE"
            break

    # --- 解析赋值 (x1=1 / true / t / 0 / false / f)
    assign_pat = re.compile(r"x\s*(\d+)\s*=\s*(true|false|t|f|1|0)", re.IGNORECASE)

    def put(k, v_str):
        v = v_str.strip().lower()
        is_true = v in ("1", "t", "true")
        model_dict[k] = k if is_true else -k

    # 优先：若某行 '=' 次数 == N_hint（若有），则以该行为整体解析
    picked_line = None
    if N_hint is not None:
        for line in lines:
            if line.count("=") == N_hint:
                picked_line = line
                break

    if picked_line is not None:
        for var_str, val_str in assign_pat.findall(picked_line):
            put(int(var_str), val_str)
    else:
        # 兜底：全文件扫描
        for line in lines:
            for var_str, val_str in assign_pat.findall(line):
                put(int(var_str), val_str)

    # 汇总 model
    model = [model_dict[k] for k in sorted(model_dict)]

    # 估计变量个数：优先 header；否则 N_hint；否则从子句或模型里推断最大变量编号
    nvars = None
    if nvars_hdr is not None:
        nvars = nvars_hdr
    elif N_hint is not None:
        nvars = N_hint
    else:
        max_from_clauses = max((abs(l) for c in clauses for l in c), default=0)
        max_from_model = max((abs(l) for l in model), default=0)
        nvars = max(max_from_clauses, max_from_model) or None

    return clauses, model, sat, nvars


def infer_N_from_filename(fname: str):
    """
    从文件名推断 N：支持 N75 / N_75 / _N_75_ / ... 形式
    """
    m = re.search(r"[ _\-]N_?(\d+)", fname)
    if m:
        return int(m.group(1))
    m2 = re.search(r"N(\d+)", fname)
    if m2:
        return int(m2.group(1))
    return None


# ================== 校验与统计 ==================
def violated_clauses(clauses, model):
    assignment = {abs(lit): (lit > 0) for lit in model}
    unsatisfied = []
    for idx, clause in enumerate(clauses):
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, None)
            if val is not None:
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
        if not satisfied:
            unsatisfied.append((idx, clause))
    return unsatisfied, assignment


def check_model_with_minisat(clauses, model):
    with Minisat22(bootstrap_with=clauses) as solver:
        return solver.solve(assumptions=model)


# ================== 目录枚举 ==================
def list_target_dirs():
    dirs = []

    # FOLDER1: 直接子目录中以指定前缀开头的
    if os.path.isdir(FOLDER1_ROOT):
        for name in os.listdir(FOLDER1_ROOT):
            if name.startswith(FOLDER1_PREFIX):
                full = os.path.join(FOLDER1_ROOT, name)
                if os.path.isdir(full):
                    dirs.append(full)

    # FOLDER2: 直接子目录中以指定前缀开头的
    if os.path.isdir(FOLDER2_ROOT):
        for name in os.listdir(FOLDER2_ROOT):
            if name.startswith(FOLDER2_PREFIX):
                full = os.path.join(FOLDER2_ROOT, name)
                if os.path.isdir(full):
                    dirs.append(full)

    return sorted(dirs)


# ================== 主流程 ==================
def main():
    target_dirs = list_target_dirs()
    print("[INFO] Will scan dirs:")
    for d in target_dirs:
        print("  -", d)

    summary_rows = []
    detail_rows = []

    for sub_dir in target_dirs:
        files = [f for f in os.listdir(sub_dir) if f.lower().endswith(".cnf") or f.lower().endswith(".txt")]
        files.sort()

        satisfied_number = 0
        not_satisfied_number = 0
        incomplete_assignment = 0
        total_files = 0

        # 试图从目录名中也提取 N（可选）


        for file_name in files:
            N_from_dir = infer_N_from_filename(file_name)
            total_files += 1
            fp = os.path.join(sub_dir, file_name)

            # 先从文件名估 N，再由正文 header 覆盖
            N_from_file = infer_N_from_filename(file_name)
            N_hint = N_from_file or N_from_dir

            try:
                clauses, model, sat, nvars = parse_cnf_and_model(fp, N_hint=N_hint)
            except Exception as e:
                detail_rows.append({
                    "dir": sub_dir,
                    "file": file_name,
                    "predicted": None,
                    "nvars": None,
                    "model_len": None,
                    "checked": False,
                    "satisfied": None,
                    "note": f"parse_error: {e}"
                })
                continue

            # 仅对预测为 SAT 的条目做赋值校验
            if sat == "SATISFIABLE":
                # 若没有足够完整的赋值，与之前逻辑一致：跳过统计到成功/失败；单独计数一下
                if nvars is None or len(model) != int(nvars):
                    incomplete_assignment += 1
                    detail_rows.append({
                        "dir": sub_dir,
                        "file": file_name,
                        "predicted": sat,
                        "nvars": nvars,
                        "model_len": len(model),
                        "checked": False,
                        "satisfied": None,
                        "note": "incomplete_assignment"
                    })
                    continue

                is_ok = check_model_with_minisat(clauses, model)
                if is_ok:
                    satisfied_number += 1
                    note = "assignment_satisfies_cnf"
                else:
                    not_satisfied_number += 1
                    note = "assignment_does_not_satisfy_cnf"

                detail_rows.append({
                    "dir": sub_dir,
                    "file": file_name,
                    "predicted": sat,
                    "nvars": nvars,
                    "model_len": len(model),
                    "checked": True,
                    "satisfied": bool(is_ok),
                    "note": note
                })
            else:
                # 非 SAT 或未知，记录一下，但不计入 satisfied/unsatisfied 统计
                detail_rows.append({
                    "dir": sub_dir,
                    "file": file_name,
                    "predicted": sat,
                    "nvars": nvars,
                    "model_len": len(model),
                    "checked": False,
                    "satisfied": None,
                    "note": "pred_not_sat"
                })

        # 汇总（分母仅包含“预测为 SAT 且赋值完整”的文件数）
        denom = satisfied_number + not_satisfied_number
        row = {
            "sub_dir": sub_dir,
            "N_hint": N_from_dir,
            "files_total": total_files,
            "assignments_satisfied_number": satisfied_number,
            "assignments_not_satisfied_number": not_satisfied_number,
            "assignments_checked_total": denom,
            "assignments_satisfied_rate": _safe_div(satisfied_number, denom),
            "incomplete_assignment_skipped": incomplete_assignment,
        }
        print("[SUMMARY]", row)
        summary_rows.append(row)

    # 保存 CSV
    df_sum = pd.DataFrame(summary_rows)
    df_sum = df_sum.sort_values(by=["sub_dir"])
    df_sum.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Summary saved to: {SUMMARY_CSV}")

    df_details = pd.DataFrame(detail_rows)
    df_details.to_csv(DETAILS_CSV, index=False)
    print(f"[OK] Details saved to: {DETAILS_CSV}")


if __name__ == "__main__":
    main()
