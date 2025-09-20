import os

from pysat.formula import CNF
from pysat.solvers import Glucose3
import random
import copy
from pysat.solvers import Minisat22


file_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\fix_cnf\fixed_set'
file_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set'
file_name = 'k3_N75_L412_alpha5.5_inst82_UNSAT.cnf'
file_path = os.path.join(file_dir, file_name)


# 读取原始 CNF 文件
original_cnf = CNF(from_file=file_path)
current_cnf = CNF(from_clauses=original_cnf.clauses)  # ✅ 正确复制方式

round_count = 0
max_rounds = len(current_cnf.clauses)  # 最多修复次数
print("max_rounds:", max_rounds)
while round_count < max_rounds:
    solver = Glucose3()

    # 找最大变量编号
    next_var = max(max(map(abs, clause)) for clause in current_cnf.clauses if clause) if current_cnf.clauses else 0

    assumptions = []
    mapping = {}

    for i, clause in enumerate(current_cnf.clauses):
        if not clause:
            continue
        next_var += 1
        aux = next_var
        mapping[i] = aux
        assumptions.append(aux)
        solver.add_clause(clause + [-aux])

    # 尝试求解
    if solver.solve(assumptions=assumptions):
        print(f"✔️ SAT achieved after {round_count} fix(es).")
        output_path = os.path.join(file_dir, file_name[:-4]+"_fixed.cnf")
        current_cnf.to_file(output_path)
        with Minisat22(bootstrap_with=current_cnf.clauses) as m:
            if m.solve():
                print("✅ The fixed CNF is SAT.")
            else:
                print("❌ The fixed CNF is still UNSAT.")
        print(f"Fixed CNF saved to {output_path}")
        break

    # 否则提取冲突子句并删除一条
    core_aux_vars = solver.get_core()
    clause_indices = [i for i, v in mapping.items() if v in core_aux_vars]

    if not clause_indices:
        print("⚠️ No identifiable core. Stopping.")
        break

    to_delete = random.choice(clause_indices)
    print(f"❌ Attempt {round_count+1}: Deleting clause {to_delete} from core {clause_indices}")

    current_cnf.clauses.pop(to_delete)
    round_count += 1

else:
    print("❌ Reached max attempts but CNF is still UNSAT.")
