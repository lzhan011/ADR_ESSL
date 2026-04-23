import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from random import randint, sample
from tqdm import tqdm
from pysat.solvers import Minisat22
from statistics import median

# ---------- 字体与输出设置 ----------
tnr_candidates = [
    os.path.expanduser("~/.local/share/fonts/Times New Roman.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Bold.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Italic.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Bold Italic.ttf"),
]
for p in tnr_candidates:
    if os.path.exists(p):
        fm.fontManager.addfont(p)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "FreeSerif",
    "DejaVu Serif"
]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["savefig.transparent"] = False

# ---------- 基于当前脚本的相对路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)

# 输出到 Code/Analysis_Result_Collection/phase_transition
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "..","..", "Analysis_Result_Collection", "Figure_in_paper", "phase_transition")
BASE_OUTPUT_DIR = os.path.abspath(BASE_OUTPUT_DIR)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---------- 实验参数 ----------
N_list = [75]

for N in N_list:
    alpha_values = np.arange(1.0, 10.0, 0.5)
    instances_per_alpha = 300
    k = 3

    # CNF保存目录
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"cnf_results_CDCL_N_{N}")
    os.makedirs(output_dir, exist_ok=True)

    # 图保存目录
    output_dir_figure = os.path.join(BASE_OUTPUT_DIR, "figures_CDCL_phase_transition")
    os.makedirs(output_dir_figure, exist_ok=True)

    median_branches = []
    prob_sat = []

    def generate_k_sat(n_vars, n_clauses, k):
        clauses = []
        for _ in range(n_clauses):
            vars_in_clause = sample(range(1, n_vars + 1), k)
            clause = [var if randint(0, 1) else -var for var in vars_in_clause]
            clauses.append(clause)
        return clauses

    for alpha in tqdm(alpha_values, desc="Processing L/N values"):
        L = int(alpha * N)
        branches = []
        sat_count = 0

        for i in range(instances_per_alpha):
            cnf = generate_k_sat(N, L, k)
            with Minisat22(bootstrap_with=cnf) as m:
                result = m.solve()
                stats = m.accum_stats()
                decisions = stats.get('decisions', 0)

                if result:
                    sat_count += 1
                branches.append(decisions)

                filename = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{i+1}.txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(f"c Random {k}-SAT instance\n")
                    f.write(f"c alpha = {round(alpha, 2)}, N = {N}, L = {L}, instance = {i+1}\n")
                    f.write(f"p cnf {N} {L}\n")
                    for clause in cnf:
                        f.write(' '.join(map(str, clause)) + " 0\n")
                    f.write(f"s {'SATISFIABLE' if result else 'UNSATISFIABLE'}\n")
                    f.write(f"d decisions {decisions}\n")

        median_branches.append(median(branches))
        prob_sat.append(sat_count / instances_per_alpha)

    # ---------- 画图 ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, median_branches, '--', label='Median branches', color='black')
    ax1.set_xlabel('L / N', fontsize=18)
    ax1.set_ylabel('Median number of branches', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)

    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat, ':', color='blue', label='Prob(SAT)')
    ax2.set_ylabel('Prob(SAT)', fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper left', bbox_to_anchor=(0.0, 0.85),
               frameon=True, fontsize=15)

    plt.grid(True)
    plt.tight_layout()

    basepath = os.path.join(output_dir_figure, f"Random_3-SAT_CDCL_N_{N}_median")

    plt.savefig(basepath + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(basepath + ".pdf",            bbox_inches="tight")
    plt.savefig(basepath + ".svg",            bbox_inches="tight")
    plt.savefig(basepath + ".eps",            bbox_inches="tight")

    # plt.show()