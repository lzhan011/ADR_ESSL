import os
import matplotlib.pyplot as plt
import numpy as np
from random import randint, sample
from tqdm import tqdm
from pysat.solvers import Minisat22
from statistics import mean, median



# Constants
N = 75  # number of variables
N_list = [item for item in range(3, 21)]
N_list = [20, 25, 30, 40, 50, 60, 75]
N_list = [200]
N_list = [75]
for N in N_list:
    alpha_values = np.arange(1.0, 6.0, 0.2)
    instances_per_alpha = 300
    k = 3  # k-SAT


    # Create output directory
    output_dir = "cnf_results_CDCL/cnf_results_CDCL_N_" + str(N)
    os.makedirs(output_dir, exist_ok=True)
    output_dir_figure = "cnf_results_CDCL/figures_CDCL_phase_transition"
    os.makedirs(output_dir_figure, exist_ok=True)

    mean_branches = []
    median_branches = []
    prob_sat = []

    # Generate random k-SAT CNF
    def generate_k_sat(n_vars, n_clauses, k):
        clauses = []
        for _ in range(n_clauses):
            vars_in_clause = sample(range(1, n_vars + 1), k)
            clause = [var if randint(0, 1) else -var for var in vars_in_clause]
            clauses.append(clause)
        return clauses

    # Main loop
    for alpha in tqdm(alpha_values, desc="Processing L/N values"):
        L = int(alpha * N)
        branches = []
        sat_count = 0

        for i in range(instances_per_alpha):
            cnf = generate_k_sat(N, L, k)

            with Minisat22(bootstrap_with=cnf) as m:  # Conflict-Driven Clause Learning (CDCL)
                result = m.solve()
                stats = m.accum_stats()
                decisions = stats.get('decisions', 0)
                if result:
                    sat_count += 1
                branches.append(decisions)

                # Write CNF and result to file
                filename = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{i+1}.txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(f"c Random {k}-SAT instance\n")
                    f.write(f"c alpha = {round(alpha, 2)}, N = {N}, L = {L}, instance = {i+1}\n")
                    f.write(f"p cnf {N} {L}\n")
                    for clause in cnf:
                        f.write(" ".join(map(str, clause)) + " 0\n")
                    f.write(f"s {'SATISFIABLE' if result else 'UNSATISFIABLE'}\n")
                    f.write(f"d decisions(brunch decision number:) {decisions}\n")

        mean_branches.append(mean(branches))
        median_branches.append(median(branches))
        prob_sat.append(sat_count / instances_per_alpha)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, mean_branches, label='Mean branches', color='black')
    ax1.plot(alpha_values, median_branches, '--', label='Median branches', color='black')
    ax1.set_xlabel('L / N')
    ax1.set_ylabel('Number of branches')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat, ':', color='blue', label='Prob(sat)')
    ax2.set_ylabel('Prob(sat)')

    plt.title('Random 3-SAT, CDCL, N = ' + str(N))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figure, "Random_3-SAT_CDCL_N_"+str(N)+".png"))

    plt.show()
