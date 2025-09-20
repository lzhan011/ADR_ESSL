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



import math




# ========= Edit-distance-like CNF similarity =========
from math import inf

def _clause_overlap_k(cl1: Clause, cl2: Clause) -> int:
    # 假设子句已规范化（排序去重），交集大小就是匹配上的 literal 数
    return len(set(cl1).intersection(set(cl2)))

def _hungarian_min_cost(cost):  # cost: 方阵(list[list[float]]), return (min_cost, assign)
    n = len(cost)
    u = [0.0]*(n+1)
    v = [0.0]*(n+1)
    p = [0]*(n+1)
    way = [0]*(n+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minv = [float("inf")]*(n+1)
        used = [False]*(n+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n+1):
                if not used[j]:
                    cur = cost[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(0, n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # 增广
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1]*n
    for j in range(1, n+1):
        if p[j] != 0:
            assign[p[j]-1] = j-1
    min_cost = sum(cost[i][assign[i]] for i in range(n))
    return min_cost, assign

def cnf_similarity_edit(A: CNF, B: CNF, k: int) -> float:
    """
    编辑距离风格的相似度：
      - 子句替换成本 = k - |clauseA ∩ clauseB|
      - 插入/删除子句成本 = k
      - 通过匈牙利算法做最小代价匹配（用虚拟行/列把矩阵填成方阵，虚拟项成本=k）
      - 相似度 = 1 - ED / (k * max(|A|, |B|))
    需保证 A、B 都已去重、子句已规范化（你已有 dedupe_clauses/normalize_clause）。
    """
    A = [list(normalize_clause(cl)) for cl in A]
    B = [list(normalize_clause(cl)) for cl in B]
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return 1.0
    s = max(m, n)
    # 构造 s×s 成本矩阵；真实对真实：k - overlap；含虚拟：成本 = k
    C = [[k]*s for _ in range(s)]
    for i in range(m):
        for j in range(n):
            overlap = _clause_overlap_k(A[i], B[j])
            C[i][j] = k - overlap
    ed, _ = _hungarian_min_cost(C)   # 最小编辑代价（含填充代表插/删）
    sim = 1.0 - (ed / (k * s))
    # 数值安全裁剪
    if sim < 0: sim = 0.0
    if sim > 1: sim = 1.0
    return sim








import math

EPS = 1e-9  # 放在全局也可

def t_range_for_insert(m: int, lo: float, hi: float) -> Tuple[int, int]:
    """
    目标：lo <= m/(m+t) < hi
    推导：t ∈ [ ceil(m*(1-hi)/hi), floor(m*(1-lo)/lo - EPS) ]
    返回 (t_low, t_high)，若不可行则 t_low > t_high
    """
    if hi <= 0 or lo <= 0:
        return 1, 0
    t_low  = math.ceil(m*(1.0 - hi)/hi - EPS)
    t_high = math.floor(m*(1.0 - lo)/lo - EPS)
    return t_low, t_high

def t_range_for_delete(m: int, lo: float, hi: float) -> Tuple[int, int]:
    """
    目标：lo <= 1 - t/m < hi  =>  t ∈ [ ceil(m*(1-hi) - EPS), floor(m*(1-lo) - EPS) ]
    限制 t ∈ [1, m-1]
    """
    t_low  = max(1, math.ceil(m*(1.0 - hi) - EPS))
    t_high = min(m-1, math.floor(m*(1.0 - lo) - EPS))
    return t_low, t_high

def gen_random_k_clause(n_vars: int, k: int, ban_set: set) -> Optional[Tuple[int, ...]]:
    """
    生成一个长度为 k 的非永真、非重复子句（不依赖模型）
    ban_set: 已存在子句（normalize 后）的集合
    """
    tries = 2000
    for _ in range(tries):
        # 选 k 个不同变量，每个变量只出现一次 → 不会有 x 与 ¬x 同时出现
        vars_ = random.sample(range(1, n_vars+1), k)
        lits = []
        for v in vars_:
            s = 1 if random.random() < 0.5 else -1
            lits.append(s*v)
        cl = normalize_clause(lits)
        if cl in ban_set:
            continue
        return cl
    return None

def gen_model_true_k_clause(n_vars: int, k: int, model: Dict[int,bool], ban_set: set) -> Optional[Tuple[int, ...]]:
    """
    生成在给定模型下为真的 k-子句（每个文字符号与模型一致），同时避免重复
    """
    tries = 2000
    all_vars = list(range(1, n_vars+1))
    for _ in range(tries):
        vars_ = random.sample(all_vars, k)
        lits = []
        for v in vars_:
            s = 1 if model.get(v, False) else -1
            lits.append(s*v)
        cl = normalize_clause(lits)
        if cl in ban_set:
            continue
        return cl
    return None

def fallback_variant_by_edit(n_vars:int, base:CNF, k:int, is_sat:bool, lo:float, hi:float) -> Optional[CNF]:
    """
    随机法失败后调用：
      - UNSAT: 插入 t 个非永真、非重复的 k-子句（UNSAT 保持不变）
      - SAT:   优先删除 t 个子句（SAT 保持不变）；若区间不可达，再插入 t 个“在模型下为真”的 k-子句
    返回满足区间的 CNF 或 None
    """
    base_norm = [list(normalize_clause(cl)) for cl in base]
    base_set  = set(tuple(cl) for cl in base_norm)
    m = len(base_norm)

    if not is_sat:
        # 只插入
        tL, tH = t_range_for_insert(m, lo, hi)
        for t in range(max(1, tL), tH+1):
            cnf = [cl[:] for cl in base_norm]
            ban = set(base_set)
            ok = True
            for _ in range(t):
                cl = gen_random_k_clause(n_vars, k, ban)
                if cl is None:
                    ok = False; break
                cnf.append(list(cl)); ban.add(cl)
            if not ok: continue
            # 校验
            if not all_clauses_len_eq(cnf, k): continue
            if not no_clause_has_opposite_literals(cnf): continue
            keep, _ = solve_cnf(n_vars, cnf)
            if keep != is_sat: continue
            sim = cnf_similarity_edit(base_norm, cnf, k)
            if lo <= sim < hi:
                return cnf
        return None
    else:
        # 先删除
        tL, tH = t_range_for_delete(m, lo, hi)
        for t in range(max(1, tL), tH+1):
            # 简单删前 t 个；也可随机选
            idxs = set(range(t))
            cnf = [cl for i, cl in enumerate(base_norm) if i not in idxs]
            if not all_clauses_len_eq(cnf, k): continue
            if not no_clause_has_opposite_literals(cnf): continue
            keep, _ = solve_cnf(n_vars, cnf)
            if keep != is_sat: continue
            sim = cnf_similarity_edit(base_norm, cnf, k)
            if lo <= sim < hi:
                return cnf
        # 再尝试插入“在某模型下为真”的子句
        sat, model = solve_cnf(n_vars, base_norm)
        if sat:
            tL, tH = t_range_for_insert(m, lo, hi)
            for t in range(max(1, tL), tH+1):
                cnf = [cl[:] for cl in base_norm]
                ban = set(base_set)
                ok = True
                for _ in range(t):
                    cl = gen_model_true_k_clause(n_vars, k, model, ban)
                    if cl is None:
                        ok = False; break
                    cnf.append(list(cl)); ban.add(cl)
                if not ok: continue
                if not all_clauses_len_eq(cnf, k): continue
                if not no_clause_has_opposite_literals(cnf): continue
                keep, _ = solve_cnf(n_vars, cnf)
                if keep != is_sat: continue
                sim = cnf_similarity_edit(base_norm, cnf, k)
                if lo <= sim < hi:
                    return cnf
        return None

def similarity_in_range(sim: float, lo: float, hi: float) -> bool:
    return (sim + EPS >= lo) and (sim < hi + EPS)




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




def gen_model_true_k_clause_min_overlap(n_vars:int, k:int, model:Dict[int,bool],
                                       ban_set:set, avoid_clause:Clause) -> Optional[Tuple[int,...]]:
    """在给定模型下为真，且尽量与 avoid_clause 低重合的 k-子句。"""
    tries = 2000
    avoid_vars = set(abs(x) for x in avoid_clause)
    all_vars = list(range(1, n_vars+1))
    for _ in range(tries):
        # 优先从补集选变量，减少与 avoid_clause 的交集
        pool = [v for v in all_vars if v not in avoid_vars]
        if len(pool) >= k:
            vars_ = random.sample(pool, k)
        else:
            vars_ = random.sample(all_vars, k)  # 退化：凑不够就随机
        lits = []
        for v in vars_:
            s = 1 if model.get(v, False) else -1
            lits.append(s*v)
        cl = normalize_clause(lits)
        if cl in ban_set:  # 避免重复
            continue
        return cl
    return None

def fallback_low_sat_replace(n_vars:int, base:CNF, k:int, lo:float, hi:float,
                             max_tries:int=2000) -> Optional[CNF]:
    """
    SAT 低相似度兜底：替换 q 个子句（删 q、加 q 个“模型为真”的新子句，尽量与原子句低重合）。
    目标把编辑代价 ed ≈ k*q（甚至更大/更小，取决于重合）调进 [lo, hi)。
    """
    base_norm = [list(normalize_clause(cl)) for cl in base]
    m = len(base_norm)
    sat, model = solve_cnf(n_vars, base_norm)
    if not sat:
        return None

    # 目标编辑代价范围（与替换后子句数相同，s = m）
    s = m
    min_ed = math.ceil(k * s * (1.0 - hi) - EPS)
    max_ed = math.floor(k * s * (1.0 - lo) - EPS)
    if min_ed > max_ed:
        # 区间过窄不可达，做个小放宽（只在替换内部起作用，不影响你的筛选阈值）
        min_ed, max_ed = max(1, min_ed), max(min_ed, max_ed)

    for _ in range(max_tries):
        # 以 ed/k 估计替换个数 q，并随机抖动
        target_ed = random.randint(max(1, min_ed), max(min_ed, max_ed))
        q = max(1, min(m, int(round(target_ed / max(1, k)))))
        idxs = random.sample(range(m), q)

        cnf = [cl[:] for cl in base_norm]
        ban = set(tuple(cl) for cl in cnf)
        # 先删
        for i in sorted(idxs, reverse=True):
            del cnf[i]
        # 再加：尽量“低重合”的模型为真子句
        added = []
        ok = True
        for i in idxs:
            avoid = base_norm[i]
            cl = gen_model_true_k_clause_min_overlap(n_vars, k, model, ban, avoid)
            if cl is None:
                ok = False; break
            cnf.append(list(cl)); ban.add(cl); added.append(cl)
        if not ok:
            continue

        # 校验 & 评估
        if not all_clauses_len_eq(cnf, k): continue
        if not no_clause_has_opposite_literals(cnf): continue
        keep, _ = solve_cnf(n_vars, cnf)
        if not keep:  # SAT 应保持 SAT
            continue
        sim = cnf_similarity_edit(base_norm, cnf, k)
        if lo <= sim < hi:
            return cnf
    return None


def random_perm_local(n_vars:int, swaps:int=2) -> Dict[int,int]:
    perm = list(range(1, n_vars+1))
    for _ in range(max(0, swaps)):
        i, j = random.randrange(n_vars), random.randrange(n_vars)
        perm[i], perm[j] = perm[j], perm[i]
    return {i+1: perm[i] for i in range(n_vars)}

def random_sign_map_local(n_vars:int, flips:int=2) -> Dict[int,int]:
    smap = {i: 1 for i in range(1, n_vars+1)}
    flips = max(0, min(flips, n_vars))
    if flips > 0:
        for v in random.sample(range(1, n_vars+1), flips):
            smap[v] = -1
    return smap

def apply_perm_sign(cnf:CNF, perm:Dict[int,int], smap:Dict[int,int]) -> CNF:
    out = []
    for cl in cnf:
        mapped = []
        for lit in cl:
            v = abs(lit); s = 1 if lit > 0 else -1
            mapped.append((s * smap[v]) * perm[v])
        out.append(mapped)
    return dedupe_clauses([list(normalize_clause(cl)) for cl in out])

def fallback_low_unsat_perm_insert(n_vars:int, base:CNF, k:int, lo:float, hi:float,
                                   max_tries:int=3000) -> Optional[CNF]:
    """
    UNSAT 低相似度兜底：全局轻扰动（置换/极性翻转）后，再插入 t 个新子句，调到 [lo,hi)。
    """
    base_norm = [list(normalize_clause(cl)) for cl in base]
    m = len(base_norm)

    # 先计算仅“插入”的 t 范围；若为空也没关系，我们再靠置换带来的额外代价来补
    tL, tH = t_range_for_insert(m, lo, hi)
    t_candidates = []
    if tL <= tH:
        t_candidates = list(range(max(1, tL), tH+1))
    else:
        # 取靠近区间边界的 t 作尝试
        t_candidates = [max(1, math.ceil(m*(1.0 - lo)/lo - EPS))]

    for _ in range(max_tries):
        # 轻微扰动（不会改变 UNSAT）
        perm = random_perm_local(n_vars, swaps=2)
        smap = random_sign_map_local(n_vars, flips=2)
        pert = apply_perm_sign(base_norm, perm, smap)

        # 插入 t 个非永真、非重复子句
        t = random.choice(t_candidates)
        ban = set(tuple(cl) for cl in pert)
        cnf = [cl[:] for cl in pert]
        ok_add = True
        for _ in range(t):
            cl = gen_random_k_clause(n_vars, k, ban)
            if cl is None:
                ok_add = False; break
            cnf.append(list(cl)); ban.add(cl)
        if not ok_add:
            continue

        # 校验 & 评估
        if not all_clauses_len_eq(cnf, k): continue
        if not no_clause_has_opposite_literals(cnf): continue
        keep, _ = solve_cnf(n_vars, cnf)
        if keep:  # UNSAT 应保持 UNSAT（False）
            continue
        sim = cnf_similarity_edit(base_norm, cnf, k)
        if lo <= sim < hi:
            return cnf
    return None

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
        ("high",   0.90, 0.9999),
        ("medium", 0.80, 0.90),
        ("low",    0.70, 0.80),
    ]
    makers = {
        "high":   lambda: make_variant_high(n_vars, base, k),
        "medium": lambda: make_variant_medium(n_vars, base, k),
        "low":    lambda: make_variant_low(n_vars, base, k),
    }



    summary = []
    for level, lo, hi in targets:
        print(f"\n== Generating level: {level}  target [{lo:.2f},{hi:.4f}) ==")
        cnt, attempts = 0, 0
        max_attempts = 500 * PER_LEVEL
        while cnt < PER_LEVEL and attempts < max_attempts:
            attempts += 1
            var_cnf = makers[level]()
            var_cnf = dedupe_clauses(var_cnf)

            # A) 固定 k
            if not all_clauses_len_eq(var_cnf, k): continue
            # B) 禁永真
            if not no_clause_has_opposite_literals(var_cnf): continue
            # C) 可满足性保持
            ok, _ = solve_cnf(n_vars, var_cnf)
            if ok != is_sat: continue
            # D) 编辑距离相似度
            sim = cnf_similarity_edit(base, var_cnf, k)   # ← 不要硬编码 k=3
            if similarity_in_range(sim, lo, hi):
                out_path = os.path.join(OUT_DIR, f"{file_name[:-4]}_{level}_{cnt+1}_sim{sim:.3f}.cnf")
                write_dimacs(out_path, n_vars, var_cnf)
                print(f"[OK] {level} #{cnt+1}: similarity={sim:.3f} -> {out_path}")
                summary.append((level, cnt+1, sim, out_path))
                cnt += 1

        # 兜底：若随机法没凑够，改用精确/增强兜底
        while cnt < PER_LEVEL:
            if level == "low":
                # 低档增强：
                if is_sat:
                    fb = fallback_low_sat_replace(n_vars, base, k, lo, hi)
                else:
                    fb = fallback_low_unsat_perm_insert(n_vars, base, k, lo, hi)
            else:
                # 高/中：沿用原兜底（插/删 t）
                fb = fallback_variant_by_edit(n_vars, base, k, is_sat, lo, hi)

            if fb is None:
                # 友好提示：有可能区间与离散可达值不对齐
                m = len(base)
                if is_sat:
                    grid = [1 - t / m for t in range(1, max(2, m))]
                else:
                    grid = [m / (m + t) for t in range(1, min(10, 1 + 3 * m))]
                near = min(grid, key=lambda x: min(abs(x - lo), abs(x - hi)))
                print(f"[FAIL] level {level}: 目标区间 [{lo:.4f},{hi:.4f}) 可能与离散可达值不对齐。"
                      f" 最近可达值≈{near:.4f}。可放宽区间或增加扰动。")
                break

            sim = cnf_similarity_edit(base, fb, k)
            out_path = os.path.join(OUT_DIR, f"{file_name[:-4]}_{level}_{cnt + 1}_sim{sim:.3f}.cnf")
            write_dimacs(out_path, n_vars, fb)
            tag = "OK-FB+" if level == "low" else "OK-FB"
            print(f"[{tag}] {level} #{cnt + 1}: similarity={sim:.3f} -> {out_path}")
            summary.append((level, cnt + 1, sim, out_path))
            cnt += 1

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

    # process_one_file("example.cnf", OUT_DIR, IN_PATH, PER_LEVEL, RANDOM_SEED)
    # exit()

    PER_LEVEL = 1  # 每档生成数量
    RANDOM_SEED = 902  # 随机种子

    c_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3'
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
