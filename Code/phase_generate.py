import os, csv, random
from pysat.formula import CNF
from pysat.solvers import Glucose3


n               = 100            # The number of literals
k               = 3              # k-SAT
samples_per_m   = 20             # The number of instances
m_values        = range(300, 510, 10)
out_dir         = "phase_out"
csv_path        = "phase_stats.csv"
# ---------------------------

os.makedirs(f"{out_dir}/sat",   exist_ok=True)
os.makedirs(f"{out_dir}/unsat", exist_ok=True)

def rand_clause(n, k):
    vars_ = random.sample(range(1, n+1), k)
    return [v if random.random() < .5 else -v for v in vars_]

def write_dimacs(n, clauses, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {n} {len(clauses)}\n")
        for cls in clauses:
            f.write(" ".join(map(str, cls)) + " 0\n")

stats = []

for m in m_values:
    sat_cnt = unsat_cnt = 0
    alpha   = m / n

    for idx in range(samples_per_m):
        clauses = [rand_clause(n, k) for _ in range(m)]
        tmp = "tmp.cnf"
        write_dimacs(n, clauses, tmp)

        cnf    = CNF(from_file=tmp)
        solver = Glucose3()
        solver.append_formula(cnf.clauses)
        is_sat = solver.solve()
        solver.delete()

        # -------- 修正后的分支 --------
        if is_sat:
            category = "sat"
            sat_cnt += 1
        else:
            category = "unsat"
            unsat_cnt += 1
        # -----------------------------

        os.replace(tmp, f"{out_dir}/{category}/m{m}_{idx}.cnf")

    ratio = sat_cnt / samples_per_m
    stats.append((alpha, ratio))
    print(f"m={m:<4} α={alpha:.2f}  SAT={sat_cnt}/{samples_per_m}")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([("alpha", "sat_ratio"), *stats])

print(f"\n✓ 统计结果已写入 {csv_path}")
