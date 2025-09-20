import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from random import randint, sample
from tqdm import tqdm
from pysat.solvers import Minisat22
from statistics import median

# ---------- 字体与输出设置 ----------
# 尝试注册 Times New Roman（若你已将 TTF 放在 ~/.local/share/fonts/）
tnr_candidates = [
    os.path.expanduser("~/.local/share/fonts/Times New Roman.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Bold.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Italic.ttf"),
    os.path.expanduser("~/.local/share/fonts/Times New Roman Bold Italic.ttf"),
]
for p in tnr_candidates:
    if os.path.exists(p):
        fm.fontManager.addfont(p)

# 字体优先级：TNR -> 替代 -> DejaVu Serif
plt.rcParams["font.family"] = ["Times New Roman",
                               "Nimbus Roman No9 L",
                               "Liberation Serif",
                               "FreeSerif",
                               "DejaVu Serif"]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["legend.framealpha"] = 1.0       # 关闭半透明，避免 EPS 报警
plt.rcParams["savefig.transparent"] = False

# ---------- 实验参数 ----------
N_list = [75]
for N in N_list:
    alpha_values = np.arange(1.0, 10.0, 0.5)
    instances_per_alpha = 300
    k = 3  # k-SAT

    # 输出目录
    output_dir = f"cnf_results_CDCL/cnf_results_CDCL_N_{N}"
    os.makedirs(output_dir, exist_ok=True)
    output_dir_figure = "cnf_results_CDCL/figures_CDCL_phase_transition"
    os.makedirs(output_dir_figure, exist_ok=True)

    median_branches = []
    prob_sat = []

    # 生成随机 k-SAT CNF
    def generate_k_sat(n_vars, n_clauses, k):
        clauses = []
        for _ in range(n_clauses):
            vars_in_clause = sample(range(1, n_vars + 1), k)
            clause = [var if randint(0, 1) else -var for var in vars_in_clause]
            # 可选：避免子句中同变量重复或永真子句
            # if len({abs(x) for x in clause}) < k or any(abs(x) in [abs(y) for y in clause if y != x] and x == -y for x in clause for y in clause):
            #     continue
            clauses.append(clause)
        return clauses

    # 主循环
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

                # 记录实例（可选）
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

    # 图例：左上角下移一点，避免遮挡
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper left', bbox_to_anchor=(0.0, 0.85),
               frameon=True, fontsize=15)

    plt.grid(True)
    plt.tight_layout()

    # 多格式保存（无透明）
    basepath = os.path.join(output_dir_figure, f"Random_3-SAT_CDCL_N_{N}_median")
    plt.savefig(basepath + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(basepath + ".pdf",            bbox_inches="tight")
    plt.savefig(basepath + ".svg",            bbox_inches="tight")
    plt.savefig(basepath + ".eps",            bbox_inches="tight")

    plt.show()
