# import os
#
# from pysat.solvers import Solver
# from pysat.formula import CNF
#
# # Define the CNF formula based on the user's input
# cnf = CNF()
# clauses = [
#     [5, 3, 6],
#     [-8, -6, -2],
#     [8, 6, 4],
#     [-6, -4, -7],
#     [1, -3, 5],
#     [-5, 7, 6],
#     [7, -5, 3],
#     [5, 2, 3],
#     [8, -3, -1],
#     [2, -5, -8],
#     [-7, -5, -4],
#     [7, -3, -6],
#     [-8, -1, -2],
#     [-1, 7, 4],
#     [-8, -4, 1],
#     [4, 2, 5],
#     [-5, 4, -7],
#     [-2, 8, -6],
#     [7, 4, 8],
#     [-1, 5, 6],
#     [-6, -3, -1],
#     [8, 1, 5],
#     [-5, -1, 8],
#     [-4, 5, 7],
#     [2, -8, 1],
#     [2, -4, -1],
#     [-6, 2, 5],
#     [7, 5, 4]
# ]
# cnf.extend(clauses)
#
#
# def read_dimacs(filepath):
#     clauses = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             if line.startswith('c') or line.startswith('p'):
#                 continue
#             parts = line.strip().split()
#             if parts and parts[-1] == '0':
#                 clause = list(map(int, parts[:-1]))
#                 clauses.append(clause)
#     return clauses
#
# def cnf_to_prompt(clauses):
#     lines = []
#     for clause in clauses:
#         parts = []
#         for lit in clause:
#             parts.append(f"x{lit}" if lit > 0 else f"¬x{-lit}")
#         lines.append("(" + " ∨ ".join(parts) + ")")
#     return "\n".join(lines)
# # Use a SAT solver to determine satisfiability
#
# filepath = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_traditional_methond\unsat_cnf_low_alpha_N_8_openai_prediction\cnf_k3_N8_L28_alpha3.5_inst320.txt'
#
# dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_traditional_methond\unsat_cnf_low_alpha_N_8_openai_prediction'
# for file in os.listdir(dir):
#     filepath = os.path.join(dir, file)
#     clauses = read_dimacs(filepath)
#     # clauses = cnf_to_prompt(clauses)
#     cnf.extend(clauses)
#     with Solver(bootstrap_with=cnf.clauses) as solver:
#         satisfiable = solver.solve()
#
#     print(satisfiable)



import os

parent_folder = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_traditional_methond'
parent_folder = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code'

from pathlib import Path

def count_files_in_dir_and_subdirs(root_folder: Path):
    for folder in root_folder.rglob('*'):
        if folder.is_dir():
            file_count = sum(1 for f in folder.iterdir() if f.is_file())
            print(f"{folder}: {file_count} files")

# 示例使用
root = Path(parent_folder)
count_files_in_dir_and_subdirs(root)


