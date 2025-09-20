from pysat.formula import CNF
from pysat.solvers import Glucose3
import random

# 读取原始 CNF 文件
cnf = CNF(from_file="example.cnf")

solver = Glucose3()

# 找到当前最大变量编号，避免使用 0
next_var = max(max(map(abs, clause)) for clause in cnf.clauses if clause) if cnf.clauses else 0

assumptions = []
mapping = {}  # 映射子句编号 -> 辅助变量编号

# 包装每条子句为：(clause ∨ ¬aux)
for i, clause in enumerate(cnf.clauses):
    if not clause:
        continue  # 跳过空子句

    next_var += 1
    aux = next_var
    mapping[i] = aux
    assumptions.append(aux)
    solver.add_clause(clause + [-aux])

# 使用带 assumptions 的解算器提取 UNSAT Core
if not solver.solve(assumptions=assumptions):
    core_aux_vars = solver.get_core()  # 得到引起冲突的辅助变量集合
    print(f"UNSAT core includes {len(core_aux_vars)} auxiliary vars")

    # 获取这些辅助变量对应的子句编号
    clause_indices = [i for i, v in mapping.items() if v in core_aux_vars]
    print(f"Problematic clause indices: {clause_indices}")

    # 随机删除一条冲突子句（避免全删）
    if clause_indices:
        to_delete = set([random.choice(clause_indices)])
    else:
        to_delete = set()

    print(f"Deleting clause(s): {to_delete}")

    # 创建新 CNF（删除目标子句）
    fixed_cnf = CNF()
    for i, clause in enumerate(cnf.clauses):
        if i not in to_delete:
            fixed_cnf.append(clause)

    # 保存修复后的 CNF
    fixed_cnf.to_file("example_fixed.cnf")
    print("Fixed CNF saved to 'example_fixed.cnf'")
else:
    print("Input CNF is already satisfiable.")
