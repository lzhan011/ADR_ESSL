#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os, random, copy
from collections import Counter
from typing import List, Tuple, Optional, Dict

Clause = List[int]
CNF = List[Clause]

# ---------- DIMACS IO ----------
# ---------- DIMACS IO (robust) ----------
from typing import List, Tuple
Clause = List[int]
CNF = List[Clause]

def read_dimacs(path: str) -> Tuple[int, CNF]:
    n_vars = 0
    n_clauses_decl = 0
    clauses: CNF = []
    cur: Clause = []
    seen_p = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("c"):
                # comment line
                continue
            if not seen_p:
                # wait for 'p cnf n m'
                if line.startswith("p"):
                    parts = line.split()
                    # 形如: p cnf <n_vars> <n_clauses>
                    if len(parts) >= 4 and parts[1].lower() == "cnf":
                        n_vars = int(parts[2])
                        n_clauses_decl = int(parts[3])
                        seen_p = True
                # 在 'p' 之前出现的其他行一律忽略
                continue

            # 如果已经读够 m 个子句，忽略后续所有内容
            if n_clauses_decl and len(clauses) >= n_clauses_decl:
                break

            # 解析子句（允许跨行），以 0 作为一个子句的结束符
            tokens = line.split()
            skip_line = False
            for tok in tokens:
                if tok == "0":
                    # 结束一个子句
                    clauses.append(cur)
                    cur = []
                    # 如果已读满，则提前结束
                    if n_clauses_decl and len(clauses) >= n_clauses_decl:
                        break
                    continue
                try:
                    lit = int(tok)
                except ValueError:
                    # 这一行包含非整数（如 'UNSATISFIABLE'），整行忽略
                    cur = []  # 丢弃本行已收集的半句
                    skip_line = True
                    break
                else:
                    cur.append(lit)
            if skip_line:
                continue

    # 若文件规范，读满 m 个子句；若没有读满（文件不规范），也返回已读到的子句
    return n_vars, clauses


def write_dimacs(path: str, n_vars: int, clauses: CNF):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(x) for x in cl) + " 0\n")

# ---------- 基础工具（规范化/校验/去重） ----------
def normalize_clause(cl: Clause) -> Tuple[int, ...]:
    # 子句内排序；同时去掉重复 literal（保持最简）
    seen = set()
    out = []
    for lit in sorted(cl):
        if lit not in seen:
            out.append(lit)
            seen.add(lit)
    return tuple(out)

def has_tautology_literals(cl: Clause) -> bool:
    S = set(cl)
    return any((lit in S) and (-lit in S) for lit in S)

def dedupe_clauses(cnf: CNF) -> CNF:
    """忽略子句内顺序；相同子句只保留一份；且移除子句内重复 literal。"""
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

def all_clauses_len_eq(cnf: CNF, k: int) -> bool:
    return all(len(cl) == k for cl in cnf)

def no_clause_has_opposite_literals(cnf: CNF) -> bool:
    return all(not has_tautology_literals(cl) for cl in cnf)

# ---------- 简洁 DPLL（UCP+纯文字） ----------
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

# ---------- 相似度 ----------
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

# ---------- 变量置换 + 极性翻转（不会引入永真子句） ----------
def op_shuffle_same_k(cnf: CNF) -> CNF:
    c = [list(normalize_clause(cl)) for cl in cnf]  # 规范化子句
    random.shuffle(c)
    return c

def op_variable_permutation(n_vars: int, cnf: CNF, perm: Dict[int,int]) -> CNF:
    def map_var(v): return perm[v]
    out = []
    for cl in cnf:
        mapped = []
        for lit in cl:
            s = 1 if lit > 0 else -1
            mapped.append(s * map_var(abs(lit)))
        out.append(mapped)
    # 规范化并去重
    out = [list(normalize_clause(cl)) for cl in out]
    out = dedupe_clauses(out)
    return out

def op_polarity_flip(n_vars: int, cnf: CNF, sign_map: Dict[int,int]) -> CNF:
    """对变量做整体极性翻转：lit -> sign_map[var] * lit"""
    out = []
    for cl in cnf:
        mapped = []
        for lit in cl:
            s = 1 if lit > 0 else -1
            v = abs(lit)
            mapped.append((s * sign_map[v]) * v)
        out.append(mapped)
    # 规范化并去重
    out = [list(normalize_clause(cl)) for cl in out]
    out = dedupe_clauses(out)
    return out

def random_perm(n_vars: int, intensity: float) -> Dict[int,int]:
    base = list(range(1, n_vars+1))
    if intensity <= 0: return {i:i for i in base}
    perm = base[:]
    swaps = max(1, int(intensity * n_vars))
    for _ in range(swaps):
        i, j = random.randrange(n_vars), random.randrange(n_vars)
        perm[i], perm[j] = perm[j], perm[i]
    return {i+1: perm[i] for i in range(n_vars)}

def random_sign_map(n_vars: int, intensity: float) -> Dict[int,int]:
    """intensity∈[0,1]，约有 intensity 比例的变量被翻转符号"""
    smap = {i: 1 for i in range(1, n_vars+1)}
    flips = max(0, int(round(intensity * n_vars)))
    idxs = random.sample(range(1, n_vars+1), flips) if flips > 0 else []
    for v in idxs:
        smap[v] = -1
    return smap

# ---------- 变体生成（严格保持 k、无重复、无永真） ----------
def make_variant_high(n_vars:int, cnf:CNF, k:int) -> Optional[CNF]:
    c = op_shuffle_same_k(cnf)
    # 轻微极性翻转或轻微置换
    if random.random() < 0.5:
        smap = random_sign_map(n_vars, intensity=0.2)
        c = op_polarity_flip(n_vars, c, smap)
    if random.random() < 0.5:
        perm = random_perm(n_vars, intensity=0.2)
        c = op_variable_permutation(n_vars, c, perm)
    c = [list(normalize_clause(cl)) for cl in c]
    c = dedupe_clauses(c)
    return c

def make_variant_medium(n_vars:int, cnf:CNF, k:int) -> Optional[CNF]:
    perm = random_perm(n_vars, intensity=0.4)
    c = op_variable_permutation(n_vars, cnf, perm)
    smap = random_sign_map(n_vars, intensity=0.4)
    c = op_polarity_flip(n_vars, c, smap)
    c = op_shuffle_same_k(c)
    c = dedupe_clauses(c)
    return c

def make_variant_low(n_vars:int, cnf:CNF, k:int) -> Optional[CNF]:
    # 强扰动：大的变量置换 + 较多极性翻转
    perm = random_perm(n_vars, intensity=1.0)
    c = op_variable_permutation(n_vars, cnf, perm)
    smap = random_sign_map(n_vars, intensity=0.8)
    c = op_polarity_flip(n_vars, c, smap)
    c = op_shuffle_same_k(c)
    c = dedupe_clauses(c)
    return c

# ---------- 主流程 ----------
def ensure_dir(d):
    if not os.path.isdir(d): os.makedirs(d, exist_ok=True)

def process_one_file(IN_PATH, OUT_DIR, file_name, PER_LEVEL, RANDOM_SEED):
    random.seed(RANDOM_SEED)

    n_vars, base = read_dimacs(IN_PATH)
    base = dedupe_clauses(base)  # 输入先去重
    uniform, k = validate_k_uniform(base)
    if not uniform:
        print("[ERROR] 原 CNF 不是统一的 k-SAT（子句长度不一致）。当前脚本仅在统一 k 的场景下强制同长度。")
        return
    if not no_clause_has_opposite_literals(base):
        print("[ERROR] 原 CNF 自身包含永真子句（同一子句内含 x 与 -x）。请先清理后再生成。")
        return
    print(f"Loaded CNF: vars={n_vars}, clauses={len(base)} (deduped), k={k}")

    is_sat, _ = solve_cnf(n_vars, base)
    print("Original SAT? ->", is_sat)

    ensure_dir(OUT_DIR)

    targets = [
        ("high",   0.80, 0.95),
        ("medium", 0.55, 0.80),
        ("low",    0.25, 0.55),
    ]
    makers = {
        "high":   lambda: make_variant_high(n_vars, base, k),
        "medium": lambda: make_variant_medium(n_vars, base, k),
        "low":    lambda: make_variant_low(n_vars, base, k),
    }

    summary = []
    for level, lo, hi in targets:
        print(f"\n== Generating level: {level}  target [{lo:.2f},{hi:.2f}) ==")
        cnt, attempts = 0, 0
        max_attempts = 500 * PER_LEVEL
        while cnt < PER_LEVEL and attempts < max_attempts:
            attempts += 1
            var_cnf = makers[level]()
            var_cnf = dedupe_clauses(var_cnf)

            # A) 保持每个子句长度 == k
            if not all_clauses_len_eq(var_cnf, k):
                continue
            # B) 子句内不得出现互为相反的 literal，也不允许同一 literal 重复
            if not no_clause_has_opposite_literals(var_cnf):
                continue

            # C) 验证 SAT/UNSAT 结论一致
            ok, _ = solve_cnf(n_vars, var_cnf)
            if ok != is_sat:
                continue

            # D) 相似度筛选
            sim = cnf_similarity(base, var_cnf)
            if lo <= sim < hi:
                out_path = os.path.join(OUT_DIR, f"{file_name[:-4]}_{level}_{cnt+1}_sim{sim:.3f}.cnf")
                write_dimacs(out_path, n_vars, var_cnf)
                print(f"[OK] {level} #{cnt+1}: similarity={sim:.3f} -> {out_path}")
                summary.append((level, cnt+1, sim, out_path))
                cnt += 1

        if cnt < PER_LEVEL:
            print(f"[WARN] only {cnt}/{PER_LEVEL} generated for level {level}. "
                  f"可提高 max_attempts 或放宽目标区间。")

    if summary:
        print("\n=== Summary ===")
        print(f"{'level':<8} {'idx':<4} {'sim':<8} path")
        for level, idx, sim, path in summary:
            print(f"{level:<8} {idx:<4} {sim:<8.3f} {path}")
    else:
        print("\n[FAIL] 未生成任何满足条件的变体。请放宽阈值或增加尝试次数。")





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
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_1'
    original_cnf_dir = ['unsat_cnf_low_alpha_N_5_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_8_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_10_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_25_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_50_openai_prediction_o1',
                        'unsat_cnf_low_alpha_N_60_openai_prediction_o1']

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
