import os
import re
import numpy as np
from statistics import mean, median
import matplotlib.pyplot as plt

# 参数设定
output_dir = "cnf_results_openai"
N = 75
alpha_values = np.arange(3.0, 6.0, 0.5)
instances_per_alpha = 20  # 每个 alpha 下的实例数量

# 初始化结果容器
mean_branches = []
median_branches = []
prob_sat = []
avg_times = []

# 提取信息的函数
def extract_info_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    time_match = re.search(r"solve time: ([\d.]+)", content)
    branch_match = re.search(r"branches.*?(\d+)", content, re.IGNORECASE)
    conflict_match = re.search(r"conflicts.*?(\d+)", content, re.IGNORECASE)
    sat = "unsatisfiable" not in content.lower()

    time_val = float(time_match.group(1)) if time_match else 0.0
    branches = int(branch_match.group(1)) if branch_match else 0
    conflicts = int(conflict_match.group(1)) if conflict_match else 0
    return time_val, branches, conflicts, sat

# 按 alpha 值顺序处理文件
for alpha in alpha_values:
    time_list = []
    branches_list = []
    sat_count = 0

    for inst_idx in range(1, instances_per_alpha + 1):
        fname = f"cnf_k3_N{N}_L{int(alpha*N)}_alpha{round(alpha,2)}_inst{inst_idx}.txt"
        filepath = os.path.join(output_dir, fname)
        if not os.path.exists(filepath):
            print(f"[Warning] Missing file: {fname}")
            continue

        time_val, branches, conflicts, sat = extract_info_from_file(filepath)
        time_list.append(time_val)
        branches_list.append(branches)
        if sat:
            sat_count += 1

    # 统计结果
    if branches_list:
        mean_branches.append(mean(branches_list))
        median_branches.append(median(branches_list))
        avg_times.append(mean(time_list))
        prob_sat.append(sat_count / len(branches_list))
    else:
        mean_branches.append(0)
        median_branches.append(0)
        avg_times.append(0)
        prob_sat.append(0)

# ====================
# 图 1：Mean / Median branches + Prob(sat)
# ====================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：branches
line1, = ax1.plot(alpha_values, mean_branches, label="Mean branches", color="orange", marker='o')
line2, = ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="red", marker='s')
ax1.set_xlabel("L / N")
ax1.set_ylabel("Number of branches")
ax1.set_ylim(0, max(mean_branches + median_branches) * 1.2)

# 添加数值标签
for x, y in zip(alpha_values, mean_branches):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
for x, y in zip(alpha_values, median_branches):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# 右轴：prob(sat)
ax2 = ax1.twinx()
line3, = ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="blue", marker='^')
ax2.set_ylabel("Prob(sat)")
ax2.set_ylim(0, 1.05)

# 添加数值标签
for x, y in zip(alpha_values, prob_sat):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# 合并图例
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.title("GPT-3.5 on Random 3-SAT (Branches & SAT Probability)")
plt.grid(True)
plt.tight_layout()
plt.show()



# ====================
# 图 2：Avg solve time
# ====================
fig2, ax = plt.subplots(figsize=(10, 5))

line4, = ax.plot(alpha_values, avg_times, '-o', label="Avg solve time (s)", color="green")
ax.set_xlabel("L / N")
ax.set_ylabel("Avg solve time (seconds)")
ax.set_ylim(0, max(avg_times) * 1.2)
ax.set_title("GPT-3.5 on Random 3-SAT (Solve Time Only)")
ax.legend(loc="upper center")

# 添加数值标签
for x, y in zip(alpha_values, avg_times):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

plt.grid(True)
plt.tight_layout()
plt.show()
