import os
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
from pysat.solvers import Minisat22

# === 设置文件路径 ===
# file_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\fix_cnf\fixed_set'

import os
from pysat.formula import CNF, WCNF, IDPool
from pysat.examples.rc2 import RC2



def parse_cnf(filepath):
    clauses = []
    model_dict = {}

    print(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    read_clause = False
    for line in lines:
        if line.startswith("p cnf"):
            read_clause = True
            continue

        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)
    return clauses

# === 设置文件路径 ===
# file_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set'
file_dir_root =  '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
file_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL'
for sub_dir in os.listdir(file_dir_root):
    file_dir = os.path.join(file_dir_root, sub_dir)
    if not file_dir.endswith('25')  :
        continue
    for file_name in os.listdir(file_dir):
        print("file_name:", file_name)
        # file_name = 'k3_N75_L412_alpha5.5_inst82_UNSAT.cnf'
        file_path = os.path.join(file_dir, file_name)

        # === 读取原始 CNF ===
        # original_cnf = CNF(from_file=file_path)
        clauses = parse_cnf(file_path)
        original_cnf = CNF(from_clauses=clauses)
        # === 构造 WCNF，手动为每条子句添加放松变量 ===
        wcnf = WCNF()
        vpool = IDPool(start_from=original_cnf.nv + 1)  # 保证新变量编号不会与已有变量冲突
        rvar_list = []

        for clause in original_cnf.clauses:
            rvar = vpool.id()  # 创建新的放松变量
            wcnf.append(clause + [rvar], weight=1)  # clause ∨ rvar
            rvar_list.append(rvar)

        # === 使用 RC2 求解 ===
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            print(f"✔️ SAT achieved by deleting {rc2.cost} clause(s) out of {len(original_cnf.clauses)}.")

            fixed_cnf = CNF()
            deleted_clause_indices = set()

            for i, (clause, rvar) in enumerate(zip(original_cnf.clauses, rvar_list)):
                if rvar in model:  # 说明这个 clause 的 rvar 被激活，子句被“跳过”了
                    deleted_clause_indices.add(i)
                else:
                    fixed_cnf.append(clause)  # 子句被保留

            print(f"🧹 Deleted clause indices: {sorted(deleted_clause_indices)}")
            print(f"🧹 Deleted clause indices length: ", len(sorted(deleted_clause_indices)))
            # 保存修复后的 CNF 文件
            output_path = os.path.join(file_dir, file_name[:-4]+"_RC2_fixed.cnf")
            fixed_cnf.to_file(output_path)

            with Minisat22(bootstrap_with=fixed_cnf.clauses) as m:
                if m.solve():
                    print("✅ The fixed CNF is SAT.")
                else:
                    print("❌ The fixed CNF is still UNSAT.")

            print(f"✅ Fixed CNF saved to {output_path}")
