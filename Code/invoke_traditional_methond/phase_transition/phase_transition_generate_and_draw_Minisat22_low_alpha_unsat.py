import os
import matplotlib.pyplot as plt
import numpy as np
from random import randint, sample
from tqdm import tqdm
from pysat.solvers import Minisat22



# Constants

alpha_values = [3.5, 3.6, 3.7, 3.8, 3.9,4.0, 4.1, 4.2]  # 小 alpha，更难得到 UNSAT
instances_per_alpha = 10000  # 每个 alpha 生成更多实例以提高筛出 UNSAT 的可能性
k = 3  # 3-SAT


N = 150  # 固定变量数量
# Output directory
output_dir = "unsat_cnf_low_alpha_N_" + str(N)
os.makedirs(output_dir, exist_ok=True)

# 随机生成 k-SAT CNF 子句
def generate_k_sat(n_vars, n_clauses, k):
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = sample(range(1, n_vars + 1), k)
        clause = [var if randint(0, 1) else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses

# 主循环
for alpha in tqdm(alpha_values, desc="Searching for UNSAT CNFs"):
    L = int(alpha * N)
    unsat_count = 0

    for i in range(instances_per_alpha):
        cnf = generate_k_sat(N, L, k)

        with Minisat22(bootstrap_with=cnf) as m:
            result = m.solve()

            if not result:  # 如果是 UNSAT
                unsat_count += 1

                # 保存该 UNSAT 的 CNF 文件
                filename = f"UNSAT_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{i+1}.cnf"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(f"c Random {k}-SAT UNSAT instance\n")
                    f.write(f"c alpha = {round(alpha, 2)}, N = {N}, L = {L}, instance = {i+1}\n")
                    f.write(f"p cnf {N} {L}\n")
                    for clause in cnf:
                        f.write(" ".join(map(str, clause)) + " 0\n")
                    f.write(f"s UNSATISFIABLE\n")

    print(f"Alpha = {alpha}: Found {unsat_count} UNSAT instances out of {instances_per_alpha}")
