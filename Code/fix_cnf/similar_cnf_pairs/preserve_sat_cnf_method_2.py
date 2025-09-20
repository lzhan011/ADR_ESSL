#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= 配置（直接修改这里，不用命令行） =========
IN_PATH      = "cnf_k3_N5_L17_alpha3.5_inst775.txt"  # 输入 DIMACS CNF
OUT_DIR      = "variants_out"                        # 输出目录
PER_LEVEL    = 3                                     # 每个相似度档位生成的份数
RANDOM_SEED  = 12345                                 # 随机种子
# 三档相似度区间（闭区间处理，含上界；内部会做一个很小的浮点误差余量）
SIM_RANGES = [
    ("high",   0.90, 0.999),
    ("medium", 0.80, 0.89999),
    ("low",    0.70, 0.79990),
]
# 三档的局部扰动强度（绝对个数范围）：交换次数 & 极性翻转变量个数
LEVEL_STRENGTHS = {
    "high":   {"swaps": (0, 1),  "flips": (0, 1)},   # 极轻微扰动
    "medium": {"swaps": (2, 3),  "flips": (2, 3)},
    "low":    {"swaps": (5, 8),  "flips": (5, 8)},
}
# ====================================================

import os, random, copy
from collections import Counter
from typing import List, Tuple, Optional, Dict

EPS = 1e-9
Clause = List[int]
CNF = List[Clause]

# ---------- DIMACS 解析（鲁棒） ----------
def read_dimacs(path: str) -> Tuple[int, CNF]:
    n_vars = 0
    n_clauses_decl = 0
    clauses: CNF = []
    cur: Clause = []
    seen_p = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if line.startswith("c"):  # 注释行
                continue
            if not seen_p:
                if line.startswith("p"):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1].lower() == "cnf":
                        n_vars = int(parts[2])
                        n_clauses_decl = int(parts[3])
                        seen_p = True
                continue

            if n_clauses_decl and len(clauses) >= n_clauses_decl:
                break

            tokens = line.split()
            skip_line = False
            for tok in tokens:
                if tok == "0":
                    clauses.append(cur)
                    cur = []
                    if n_clauses_decl and len(clauses) >= n_clauses_decl:
                        break
                    continue
                try:
                    lit = int(tok)
                except ValueError:
                    # 该行混入非数字（如日志），整行忽略
                    cur = []
                    skip_line = True
                    break
                else:
                    cur.append(lit)
            if skip_line:
                continue
    return n_vars, clauses

def write_dimacs(path: str, n_vars: int, clauses: CNF):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(x) for x in cl) + " 0\n")

# ---------- 基础工具：规范化/检查/去重 ----------
def normalize_clause(cl: Clause) -> Tuple[int, ...]:
    # 子句内部排序、去掉重复 literal
    seen = set()
    out = []
    for lit in sorted(cl):
        if lit not in seen:
            out.append(lit)
            seen.add(lit)
    return tuple(out)

def clause_has_opposite_literals(cl: Clause) -> bool:
    S = set(cl)
    return any((lit in S) and (-lit in S) for lit in S)

def dedupe_clauses(cnf: CNF) -> CNF:
    seen = set()
    out = []
    for cl in cnf:
        key = normalize_clause(cl)
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out

def validate_k_uniform(cnf: CNF) -> Tuple[bool, Optional[int]]:
    lens = {len(cl) for cl in cnf}
    return (len(lens) == 1, next(iter(lens)) if len(lens)==1 else None)

def all_len_eq(cnf: CNF, k: int) -> bool:
    return all(len(cl) == k for cl in cnf)

def no_tautology(cnf: CNF) -> bool:
    return all(not clause_has_opposite_literals(cl) for cl in cnf)

# ---------- 简洁 DPLL（UCP + 纯文字） ----------
def unit_propagate(clauses: CNF, assignment: Dict[int,bool]):
    changed = True
    clauses = copy.deepcopy(clauses)
    assignment = dict(assignment)
    while changed:
        changed = False
        units = []
        for cl in clauses:
            if len(cl) == 0: return None, assignment
            if len(cl) == 1: units.append(cl[0])
        if not units: break
        for lit in units:
            v, val = abs(lit), (lit > 0)
            if v in assignment and assignment[v] != val:
                return None, assignment
            assignment[v] = val
            changed = True
            ncs = []
            for cl in clauses:
                if lit in cl: continue
                if -lit in cl:
                    new = [x for x in cl if x != -lit]
                    if len(new) == 0: return None, assignment
                    ncs.append(new)
                else:
                    ncs.append(cl)
            clauses = ncs
    return clauses, assignment

def pure_literal_assign(clauses: CNF, assignment: Dict[int,bool]):
    lits = set(x for cl in clauses for x in cl)
    pos = set(x for x in lits if x > 0)
    neg = set(-x for x in lits if x < 0)
    pures = []
    for v in set(abs(x) for x in lits):
        if v in pos and v not in neg: pures.append(v)
        elif v not in pos and v in neg: pures.append(-v)
    if not pures: return clauses, assignment
    clauses = copy.deepcopy(clauses)
    assignment = dict(assignment)
    for lit in pures:
        v, val = abs(lit), (lit > 0)
        assignment[v] = val
        clauses = [cl for cl in clauses if lit not in cl]
    return clauses, assignment

def choose_variable(clauses: CNF, assignment: Dict[int,bool]):
    for cl in clauses:
        for lit in cl:
            v = abs(lit)
            if v not in assignment:
                return v
    return None

def dpll(clauses: CNF, assignment: Dict[int,bool]):
    res = unit_propagate(clauses, assignment)
    if res[0] is None: return False, assignment
    clauses, assignment = res
    clauses, assignment = pure_literal_assign(clauses, assignment)
    if not clauses: return True, assignment
    v = choose_variable(clauses, assignment)
    if v is None: return True, assignment
    sat, asg = dpll(clauses + [[v]], dict(assignment))
    if sat: return True, asg
    return dpll(clauses + [[-v]], dict(assignment))

def solve_cnf(n_vars: int, clauses: CNF):
    sat, asg = dpll(clauses, {})
    if sat:
        model = dict(asg)
        for v in range(1, n_vars+1):
            model.setdefault(v, False)
        return True, model
    return False, None

# ---------- 相似度（多重集 Jaccard + 子句集合 Jaccard 加权） ----------
def literal_multiset(cnf: CNF) -> Counter:
    return Counter(l for cl in cnf for l in cl)

def jaccard_multiset(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return 1.0 if union == 0 else inter / union

def clause_set(cnf: CNF) -> set:
    return set(normalize_clause(cl) for cl in cnf)

def jaccard_set(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

def cnf_similarity(A: CNF, B: CNF, w_literals=0.6, w_clauses=0.4) -> float:
    lit_sim = jaccard_multiset(literal_multiset(A), literal_multiset(B))
    cls_sim = jaccard_set(clause_set(A), clause_set(B))
    return w_literals * lit_sim + w_clauses * cls_sim

# ---------- 近恒等映射的局部扰动：置换 & 极性翻转（绝对个数） ----------
def random_perm_local(n_vars: int, swaps: int) -> Dict[int,int]:
    perm = list(range(1, n_vars+1))
    for _ in range(max(0, swaps)):
        i, j = random.randrange(n_vars), random.randrange(n_vars)
        perm[i], perm[j] = perm[j], perm[i]
    return {i+1: perm[i] for i in range(n_vars)}

def random_sign_map_local(n_vars: int, flips: int) -> Dict[int,int]:
    smap = {i: 1 for i in range(1, n_vars+1)}
    flips = max(0, min(flips, n_vars))
    if flips > 0:
        for v in random.sample(range(1, n_vars+1), flips):
            smap[v] = -1
    return smap

def apply_perm_sign(cnf: CNF, perm: Dict[int,int], sign_map: Dict[int,int]) -> CNF:
    out = []
    for cl in cnf:
        mapped = []
        for lit in cl:
            v = abs(lit)
            s = 1 if lit > 0 else -1
            new_v = perm[v]
            new_s = s * sign_map[v]
            mapped.append(new_s * new_v)
        out.append(mapped)
    # 规范化 & 去重
    out = [list(normalize_clause(cl)) for cl in out]
    out = dedupe_clauses(out)
    return out

def shuffle_clauses(cnf: CNF) -> CNF:
    c = [list(normalize_clause(cl)) for cl in cnf]
    random.shuffle(c)
    return c

# ---------- 生成一个指定档位的变体 ----------
def sample_strength(level: str) -> Tuple[int,int]:
    s_lo, s_hi = LEVEL_STRENGTHS[level]["swaps"]
    f_lo, f_hi = LEVEL_STRENGTHS[level]["flips"]
    swaps = random.randint(s_lo, s_hi)
    flips = random.randint(f_lo, f_hi)
    return swaps, flips

def make_variant_level(n_vars:int, base:CNF, k:int, level:str) -> CNF:
    swaps, flips = sample_strength(level)
    perm = random_perm_local(n_vars, swaps=swaps)
    smap = random_sign_map_local(n_vars, flips=flips)
    c = apply_perm_sign(base, perm, smap)
    c = shuffle_clauses(c)
    return c

# ---------- 主流程 ----------
def ensure_dir(d):
    if not os.path.isdir(d): os.makedirs(d, exist_ok=True)

def similarity_in_range(sim: float, lo: float, hi: float) -> bool:
    return (sim + EPS >= lo) and (sim <= hi + EPS)

def process_one_file(IN_PATH, OUT_DIR, file, PER_LEVEL, RANDOM_SEED):
    random.seed(RANDOM_SEED)

    n_vars, base = read_dimacs(IN_PATH)
    base = dedupe_clauses(base)  # 输入先去重（若有）
    uniform, k = validate_k_uniform(base)
    if not uniform:
        print("[ERROR] 输入不是统一 k-SAT（子句长度不一致）。")
        return
    if not no_tautology(base):
        print("[ERROR] 输入含永真子句（某个子句同时含 x 与 ¬x）。请先清理。")
        return
    print(f"Loaded CNF: vars={n_vars}, clauses={len(base)} (deduped), k={k}")

    is_sat, _ = solve_cnf(n_vars, base)
    print("Original SAT? ->", is_sat)

    ensure_dir(OUT_DIR)

    summary = []
    for level, lo, hi in SIM_RANGES:
        print(f"\n== Generating level: {level}  target [{lo:.5f},{hi:.5f}] ==")
        cnt, attempts = 0, 0
        max_attempts = 1000 * PER_LEVEL  # 尝试上限，可调大
        while cnt < PER_LEVEL and attempts < max_attempts:
            attempts += 1
            var_cnf = make_variant_level(n_vars, base, k, level)
            # 强制校验
            if not all_len_eq(var_cnf, k):           # 保持 k
                continue
            if not no_tautology(var_cnf):            # 禁永真
                continue
            ok, _ = solve_cnf(n_vars, var_cnf)       # 可满足性保持
            if ok != is_sat:
                continue
            sim = cnf_similarity(base, var_cnf)      # 相似度
            if similarity_in_range(sim, lo, hi):
                out_path = os.path.join(OUT_DIR, f"{file[:-4]}_{level}_{cnt+1}_sim{sim:.6f}.cnf")
                write_dimacs(out_path, n_vars, var_cnf)
                print(f"[OK] {level} #{cnt+1}: similarity={sim:.6f} -> {out_path}")
                summary.append((level, cnt+1, sim, out_path))
                cnt += 1

        if cnt < PER_LEVEL:
            print(f"[WARN] only {cnt}/{PER_LEVEL} generated for level {level}. "
                  f"可以增大 max_attempts 或减小 swaps/flips 以更靠近高相似。")

    if summary:
        print("\n=== Summary ===")
        print(f"{'level':<8} {'idx':<4} {'similarity':<10} path")
        for level, idx, sim, path in summary:
            print(f"{level:<8} {idx:<4} {sim:<10.6f} {path}")
    else:
        print("\n[FAIL] 未生成任何满足条件的变体。请适当放宽区间或降低扰动强度。")






if __name__ == "__main__":


    # # ======== 配置（直接改这里） ========
    # IN_PATH = "example.cnf"  # 输入 DIMACS CNF
    # OUT_DIR = "variants_out"  # 输出目录
    # PER_LEVEL = 3  # 每档生成数量
    # RANDOM_SEED = 902  # 随机种子
    # # ===================================
    #
    # main()
    # exit()

    PER_LEVEL = 1  # 每档生成数量
    RANDOM_SEED = 902  # 随机种子

    c_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_2'
    original_cnf_dir = ['unsat_cnf_low_alpha_N_5_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_8_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_10_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_25_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_50_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_60_openai_prediction_o1']

    original_cnf_dir = reversed(original_cnf_dir)
    for sub_dir in original_cnf_dir:
        input_dir = os.path.join(c_root, sub_dir)
        OUT_DIR = os.path.join(output_dir_root, sub_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        for file in os.listdir(input_dir):
            input_file_path = os.path.join(input_dir, file)
            output_file_path = os.path.join(OUT_DIR, file)
            # ======== 在这里直接写参数 ========
            IN_PATH = input_file_path  # 输入 CNF 文件
            OUT_PATH = output_file_path  # 输出 CNF 文件
            process_one_file(IN_PATH, OUT_DIR, file, PER_LEVEL, RANDOM_SEED)
