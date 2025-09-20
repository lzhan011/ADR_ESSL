import os, random
from pysat.formula import CNF
from pysat.solvers import Glucose3

# ------------------ 参数 ------------------
n = 100            # 变量个数
k = 3              # k-SAT
samples = 20       # 每个 m 生成多少个实例
m_values = range(300, 510, 10)   # 子句数
out_root = "output"              # 保存目录
# -----------------------------------------

os.makedirs(f"{out_root}/sat",   exist_ok=True)
os.makedirs(f"{out_root}/unsat", exist_ok=True)

def gen_random_clause(n, k):
    """生成一个 k-literal 子句（无重复变量）"""
    vars_ = random.sample(range(1, n+1), k)
    return [v if random.random() < 0.5 else -v for v in vars_]

def write_dimacs(n, clauses, path):
    with open(path, "w") as f:
        f.write(f"p cnf {n} {len(clauses)}\n")
        for cls in clauses:
            f.write(" ".join(map(str, cls)) + " 0\n")

for m in m_values:
    sat_cnt = unsat_cnt = 0
    alpha = m / n

    for idx in range(samples):
        # 1) 生成随机公式
        clauses = [gen_random_clause(n, k) for _ in range(m)]
        tmp = "tmp.cnf"
        write_dimacs(n, clauses, tmp)

        # 2) 用 PySAT 判定 SAT/UNSAT
        cnf = CNF(from_file=tmp)
        solver = Glucose3()
        solver.append_formula(cnf.clauses)
        is_sat = solver.solve()
        solver.delete()

        # 3) 按类别保存
        cat = "sat" if is_sat else "unsat"
        if is_sat: sat_cnt += 1
        else:      unsat_cnt += 1
        dst = f"{out_root}/{cat}/m{m}_{idx}.cnf"
        os.replace(tmp, dst)

    print(f"m={m:<4} α={alpha:.2f}  SAT={sat_cnt}/{samples}  UNSAT={unsat_cnt}/{samples}")
