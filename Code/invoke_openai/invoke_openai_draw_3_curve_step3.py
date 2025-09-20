# 创建一个 Python 脚本文件，内容为使用 Minisat22 解析 CNF 并绘图的代码

import os
import re
import numpy as np
from statistics import mean, median
import matplotlib.pyplot as plt
from pysat.solvers import Minisat22

# 参数设定
output_dir = "cnf_results_openai_4.1"
alpha_values = np.arange(3.0, 6.0, 0.5)



model_selected = 'o1' # 'gpt-4-turbo' # 'chatgpt-4o-latest'  #'gpt-4.1' # 'gpt-4o' #'gpt-3.5-turbo'  'gpt-3.5-turbo-0125'    'o1'
# o3,  o4-mini,

# o1-pro, o3, o3-mini, o3-pro, o3-deep-research, o4-mini,
# 输出目录
# output_dir = "cnf_results_openai_"+model_selected + "_small_alpha"
output_dir = "cnf_results_openai_"+model_selected
# output_dir = "cnf_results_openai_gpt-4o_small_alpha"
# output_dir = "cnf_results_openai_gpt-3.5-turbo_small_alpha"
# alpha_values = np.arange(1.0, 4.5, 0.5)
alpha_values = np.arange(3.0, 6.0, 0.5)

output_dir_figure = os.path.join('figures', output_dir)
os.makedirs('figures', exist_ok=True)
os.makedirs(output_dir_figure, exist_ok=True)

N = 75

instances_per_alpha = 20

# 初始化容器
mean_branches, median_branches, prob_sat, avg_times = [], [], [], []
mean_branches_mini, median_branches_mini, prob_sat_mini = [], [], []
gt_prob_sat, gpt_vs_mini_accuracy = [], []

# 提取信息函数
def extract_info_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 初始化
    clauses = []
    read_clause = False
    time_val = 0.0
    branches = 0
    conflicts = 0
    sat = "not_decided"

    for line in lines:
        if line.startswith("p cnf"):
            read_clause = True
            continue
        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            # 解析子句
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)

        # 提取 GPT 输出信息
        if "solve time" in line.lower():
            match = re.search(r"solve time: ([\d.]+)", line, re.IGNORECASE)
            if match:
                time_val = float(match.group(1))
        if "branches" in line.lower():
            match = re.search(r"branches.*?(\d+)", line, re.IGNORECASE)
            if match:
                branches = int(match.group(1))
        if "conflicts" in line.lower():
            match = re.search(r"conflicts.*?(\d+)", line, re.IGNORECASE)
            if match:
                conflicts = int(match.group(1))
        if "satisfiable" in line.lower() and "unsatisfiable" not in line.lower():
            sat = True
        if "unsatisfiable" in line.lower():
            sat = False



    if "satisfiable" in lines[-3].lower() or "unsatisfiable" in lines[-3].lower():

        match = re.search(r".*?(\d+)", lines[-2], re.IGNORECASE)
        if match:
            branches = int(match.group(1))


        match = re.search(r"conflicts.*?(\d+)", lines[-1], re.IGNORECASE)
        if match:
            conflicts = int(match.group(1))
    else:
        branches = 'not_found'
        conflicts = 'not_found'


    return time_val, branches, conflicts, sat, clauses


# 主循环
for alpha in alpha_values:
    time_list, branches_list, sat_count = [], [], 0
    branches_list_mini = []
    mini_sat_count, correct_count = 0, 0

    for inst_idx in range(1, instances_per_alpha + 1):
        fname = f"cnf_k3_N{N}_L{int(alpha*N)}_alpha{round(alpha,2)}_inst{inst_idx}.txt"
        filepath = os.path.join(output_dir, fname)
        if not os.path.exists(filepath):
            continue
        time_val, branches, conflicts, gpt_sat, clauses = extract_info_from_file(filepath)
        if branches == 'not_found' or conflicts == 'not_found' or gpt_sat == 'not_decided':
            continue
        time_list.append(time_val)
        branches_list.append(branches)
        if gpt_sat == True:
            sat_count += 1

        # 调用 Minisat22
        with Minisat22(bootstrap_with=clauses) as m:
            mini_sat = m.solve()
            if mini_sat:
                mini_sat_count += 1
            if mini_sat == gpt_sat:
                correct_count += 1


            # mini_sat = m.solve()
            stats = m.accum_stats()
            decisions = stats.get('decisions', 0)
            # if mini_sat:
            #     sat_count += 1
            branches_list_mini.append(decisions)

    mean_branches.append(mean(branches_list) if branches_list else 0)
    median_branches.append(median(branches_list) if branches_list else 0)
    mean_branches_mini.append(mean(branches_list_mini) if branches_list_mini else 0)
    median_branches_mini.append(median(branches_list_mini) if branches_list_mini else 0)
    avg_times.append(mean(time_list) if time_list else 0)
    prob_sat.append(sat_count / len(branches_list) if branches_list else 0)
    gt_prob_sat.append(mini_sat_count / len(branches_list) if branches_list else 0)
    gpt_vs_mini_accuracy.append(correct_count / len(branches_list) if branches_list else 0)

# ===================
# 图1：GPT vs Minisat SAT 概率对比图
# ===================
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, prob_sat, '-o', label="GPT" + model_selected +" SAT Prob", color='orange')
plt.plot(alpha_values, gt_prob_sat, '-s', label="CDCL SAT Prob", color='blue')
plt.xlabel("L / N")
plt.ylabel("Probability of SAT")
plt.title("SAT Probability: GPT-"+model_selected+" vs CDCL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_figure, "plot_sat_probability_comparison.png"))
plt.show()

# ===================
# 图2：分支数量对比图
# ===================
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, mean_branches, '-o', label="Mean branches (GPT)", color='red')
plt.plot(alpha_values, median_branches, '--s', label="Median branches (GPT)", color='red')
plt.xlabel("L / N")
plt.ylabel("Branches (GPT estimated)")
plt.title("GPT" + model_selected + ": Mean & Median Branches")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_figure,"plot_branches_gpt.png"))
plt.show()

# ===================
# 图3：GPT准确率 vs Minisat
# ===================
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, gpt_vs_mini_accuracy, '-^', label="GPT Accuracy vs Minisat", color='purple')
plt.xlabel("L / N")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("GPT" + model_selected + " vs CDCL SAT Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_figure,"plot_accuracy_gpt_vs_minisat.png"))
plt.show()




# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(alpha_values, mean_branches_mini, label='Mean branches', color='black')
ax1.plot(alpha_values, median_branches_mini, '--', label='Median branches', color='black')
ax1.set_xlabel('L / N')
ax1.set_ylabel('Number of branches')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(alpha_values, gt_prob_sat, ':', color='black', label='Prob(sat)')
ax2.set_ylabel('Prob(sat)')

plt.title('Random 3-SAT, CDCL, N = 75')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_figure,'Random_3-SAT_CDCL_N_75.png'))
plt.show()


