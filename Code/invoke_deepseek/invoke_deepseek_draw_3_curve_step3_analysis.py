# 创建一个 Python 脚本文件，内容为使用 Minisat22 解析 CNF 并绘图的代码

import os
import re
import numpy as np
from statistics import mean, median
import matplotlib.pyplot as plt
from pysat.solvers import Minisat22

# 参数设定

N = 75
output_dir = "cnf_results_deepseek_N_" + str(N)
alpha_values = np.arange(3.0, 6.0, 0.5)
instances_per_alpha = 100

# 初始化容器
mean_branches, median_branches, prob_sat, avg_times = [], [], [], []
mean_branches_mini, median_branches_mini, prob_sat_mini = [], [], []
gt_prob_sat, gpt_vs_mini_accuracy = [], []

import re

import re

def extract_info_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    clauses = []
    in_clause_section = False

    for line in lines:
        if line.startswith("p cnf"):
            in_clause_section = True
            continue
        if in_clause_section:
            # 如果遇到非纯文字行，则退出子句解析
            if not re.match(r"^[-\d\s]+0$", line):  # e.g., '1 -3 5 0'
                in_clause_section = False
                continue
            try:
                clause = [int(x) for x in line.split() if x != "0"]
                clauses.append(clause)
            except ValueError:
                continue  # 忽略非整数字符行

    # 提取最后三行结果
    # 兼容有 * 或无 * 的 SAT/UNSAT 标记
    for i in range(len(lines)-1, -1, -1):
        if 'satisfiable' in lines[i].lower():
            sat_line = lines[i].lower().replace("*", "").strip()
            branches_line = lines[i + 1]
            conflicts_line = lines[i + 2]
            break
    else:
        raise ValueError("No satisfiability result found.")

    # 正则提取数字
    branches_match = re.search(r"(\d+)", branches_line)
    conflicts_match = re.search(r"(\d+)", conflicts_line)

    sat = "unsatisfiable" not in sat_line
    branches = int(branches_match.group(1)) if branches_match else 0
    conflicts = int(conflicts_match.group(1)) if conflicts_match else 0
    time_val = 0.0  # DeepSeek 不提供时间的话默认为 0

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
        time_list.append(time_val)
        branches_list.append(branches)
        if gpt_sat:
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
plt.plot(alpha_values, prob_sat, '-o', label="DeepSeek SAT Prob", color='orange')
plt.plot(alpha_values, gt_prob_sat, '-s', label="CDCL SAT Prob", color='blue')
plt.xlabel("L / N")
plt.ylabel("Probability of SAT")
plt.title("SAT Probability: DeepSeek vs CDCL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("DeepSeek_plot_sat_probability_comparison.png")


# ===================
# 图2：分支数量对比图
# ===================
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, mean_branches, '-o', label="Mean branches (DeepSeek)", color='red')
plt.plot(alpha_values, median_branches, '--s', label="Median branches (DeepSeek)", color='red')
plt.xlabel("L / N")
plt.ylabel("Branches (DeepSeek estimated)")
plt.title("DeepSeek: Mean & Median Branches")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DeepSeek_plot_branches_gpt.png")
plt.show()

# ===================
# 图3：GPT准确率 vs Minisat
# ===================
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, gpt_vs_mini_accuracy, '-^', label="DeepSeek Accuracy", color='purple')
plt.xlabel("L / N")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("DeepSeek SAT Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("DeepSeek_plot_accuracy_gpt_vs_minisat.png")
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

plt.title('Random 3-SAT, CDCL, N = 75, on DeepSeek Used instances')
plt.grid(True)
plt.tight_layout()
plt.show()


