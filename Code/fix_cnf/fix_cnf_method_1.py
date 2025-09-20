from pysat.formula import CNF
from pysat.solvers import Solver
import random
import copy

def is_satisfiable(cnf):
    with Solver(bootstrap_with=cnf.clauses) as solver:
        return solver.solve()

def fix_unsat_cnf(filename, max_attempts=50):
    # 读取原始 CNF
    original_cnf = CNF(from_file=filename)
    print(f"Original clauses: {len(original_cnf.clauses)}")

    if is_satisfiable(original_cnf):
        print("Already SAT")
        return original_cnf

    # 尝试删除部分子句修复
    for attempt in range(max_attempts):
        cnf_copy = copy.deepcopy(original_cnf)
        num_to_remove = random.randint(1, int(len(cnf_copy.clauses) * 0.2))  # 删除最多20%
        indices_to_remove = random.sample(range(len(cnf_copy.clauses)), num_to_remove)
        for idx in sorted(indices_to_remove, reverse=True):
            del cnf_copy.clauses[idx]

        if is_satisfiable(cnf_copy):
            print(f"Fixed SAT after removing {num_to_remove} clauses in attempt {attempt + 1}")
            return cnf_copy

    print("Failed to fix CNF after maximum attempts.")
    return None



fixed = fix_unsat_cnf("example.cnf")
if fixed:
    fixed.to_file("example_fixed.cnf")
