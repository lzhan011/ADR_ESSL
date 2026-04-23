import os
from pysat.formula import CNF, WCNF, IDPool
from pysat.examples.rc2 import RC2
from pysat.solvers import Minisat22



def parse_cnf(filepath):
    clauses = []

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


# Linux/HPC root directory
file_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
original_cnf_dir = [
    'unsat_cnf_low_alpha_N_5_openai_prediction_o1',
    'unsat_cnf_low_alpha_N_8_openai_prediction_o1',
    'unsat_cnf_low_alpha_N_10_openai_prediction_o1',
    'unsat_cnf_low_alpha_N_25_openai_prediction_o1',
    'unsat_cnf_low_alpha_N_50_openai_prediction_o1',
    'unsat_cnf_low_alpha_N_60_openai_prediction_o1',
]

for sub_dir in original_cnf_dir:
    file_dir = os.path.join(file_dir_root, sub_dir)
    if not os.path.isdir(file_dir):
        print(f"Skip missing directory: {file_dir}")
        continue

    for file_name in os.listdir(file_dir):
        # Only process raw CNF files; skip already fixed outputs
        if '_RC2_fixed' in file_name or '_fixed' in file_name:
            continue
        if not (file_name.endswith('.cnf') or file_name.endswith('.txt')):
            continue

        print('file_name:', file_name)
        file_path = os.path.join(file_dir, file_name)

        clauses = parse_cnf(file_path)
        original_cnf = CNF(from_clauses=clauses)

        # Build WCNF with one relaxation variable per clause
        wcnf = WCNF()
        vpool = IDPool(start_from=original_cnf.nv + 1)
        rvar_list = []

        for clause in original_cnf.clauses:
            rvar = vpool.id()
            wcnf.append(clause + [rvar], weight=1)
            rvar_list.append(rvar)

        with RC2(wcnf) as rc2:
            model = rc2.compute()
            print(f"SAT achieved by deleting {rc2.cost} clause(s) out of {len(original_cnf.clauses)}.")

            fixed_cnf = CNF()
            deleted_clause_indices = set()

            for i, (clause, rvar) in enumerate(zip(original_cnf.clauses, rvar_list)):
                if rvar in model:
                    deleted_clause_indices.add(i)
                else:
                    fixed_cnf.append(clause)

            print(f"Deleted clause indices: {sorted(deleted_clause_indices)}")
            print('Deleted clause indices length:', len(sorted(deleted_clause_indices)))

            output_path = os.path.join(file_dir, os.path.splitext(file_name)[0] + '_RC2_fixed.cnf')
            fixed_cnf.to_file(output_path)

            with Minisat22(bootstrap_with=fixed_cnf.clauses) as m:
                if m.solve():
                    print('The fixed CNF is SAT.')
                else:
                    print('The fixed CNF is still UNSAT.')

            print(f"Fixed CNF saved to {output_path}")
