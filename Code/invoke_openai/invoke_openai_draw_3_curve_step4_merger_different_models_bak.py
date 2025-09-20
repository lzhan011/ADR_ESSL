"""
multi_model_compare.py
======================

一次性比较多个大模型在随机 3‑SAT 上的表现：
  • SAT 概率
  • 平均分支数
  • 预测准确率（LLM vs MiniSat）

运行前请准备好各模型的结果目录：
  cnf_results_openai_o1/
  cnf_results_openai_o3-mini/
  cnf_results_openai_gpt-4o/
  cnf_results_openai_gpt-3.5-turbo/
"""

import os
import re
from statistics import mean, median
import numpy as np
import matplotlib.pyplot as plt
from pysat.solvers import Minisat22

# ---------- 全局设置 ----------
MODELS = ['o1', 'o3-mini', 'gpt-4o', 'gpt-3.5-turbo']
ALPHAS = np.arange(3.0, 6.0, 0.5)        # L / N
N               = 75                      # 变量数
INSTANCES_EACH  = 20                      # 每个 α 的公式数
FIG_DIR         = 'figures_comparison'
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ['gpt-3.5-turbo',  'gpt-4o',  'gpt-4.1', 'o1',]  # 'gpt-4o-latest', 'o3-mini', o3
output_dir_base = "cnf_results_openai_"
suffix = ''

# MODELS = ['gpt-4.1', 'gpt-3.5-turbo',  'gpt-4o',   'o1',]  # 'gpt-4o-latest', 'o3-mini', o3
# output_dir_base = "cnf_results_openai_"
# suffix = '_small_alpha'
# ALPHAS = np.arange(1.0, 4.5, 0.5)        # L / N





FIG_DIR  = os.path.join(FIG_DIR, output_dir_base + suffix)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------- Global plotting style for paper-sized figures ----------
plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
})


# ---------- 工具函数 ----------
def extract_info_from_file(filepath: str):
    """
    解析单个结果文件，返回
      time, branches, conflicts, gpt_sat(bool/None), clauses(list[list[int]])
    —— 你原来的 extract_info_from_file() 函数直接拿来即可。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses, read_clause = [], False
    time_val, branches, conflicts = 0.0, 0, 0
    sat = None    # True / False / None
    sat_index = None
    branches_number, conflicts_number = None, None
    for line_index in range(len(lines)):
        line = lines[line_index]
        if line.startswith("p cnf"):
            read_clause = True
            continue
        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)


        # 使用正则提取所有英文字母
        letters = re.findall(r'[a-zA-Z]', line)

        # 拼接成一个单词
        letters_word = ''.join(letters).lower()
        if letters_word == "unsatisfiable":
            sat = False
            sat_index = line_index
        elif letters_word == "satisfiable":
            sat = True
            sat_index = line_index

    if sat_index is not None:
        branches_index = sat_index + 1
        branches_number = lines[branches_index]
        branches_number = re.findall(r'\d*', branches_number)
        branches_number = int(''.join(branches_number))

        conflicts_index = sat_index + 2
        conflicts_number = lines[conflicts_index]
        conflicts_number = re.findall(r'\d*', conflicts_number)
        conflicts_number = int(''.join(conflicts_number))




    return time_val, branches_number, conflicts_number, sat, clauses


def collect_metrics_for_model(model: str):
    """
    针对一个模型，把各 α 的统计量返回：
        prob_sat_list,
        mean_branches_list,
        accuracy_list
    """
    # output_dir = f"cnf_results_openai_{model}"
    output_dir = output_dir_base + model + suffix

    prob_sat_lst, mean_br_lst, acc_lst = [], [], []

    for alpha in ALPHAS:
        branches_gpt, sat_gpt_cnt = [], 0
        branches_mini, sat_mini_cnt, correct_cnt = [], 0, 0

        for idx in range(1, INSTANCES_EACH + 1):
            fname = f"cnf_k3_N{N}_L{int(alpha*N)}_alpha{alpha:.1f}_inst{idx}.txt"
            path  = os.path.join(output_dir, fname)
            if not os.path.exists(path):
                continue

            t, br, conf, gpt_sat, clauses = extract_info_from_file(path)
            if gpt_sat is None or br is None:
                continue

            # GPT 结果
            branches_gpt.append(br)
            if gpt_sat: sat_gpt_cnt += 1

            # MiniSat 真值
            with Minisat22(bootstrap_with=clauses) as m:
                mini_sat = m.solve()
                if mini_sat: sat_mini_cnt += 1
                if mini_sat == gpt_sat:
                    correct_cnt += 1
                branches_mini.append(m.accum_stats().get('decisions', 0))

        total = len(branches_gpt) if branches_gpt else 1  # 避免除零
        prob_sat_lst.append(sat_gpt_cnt / total)
        mean_br_lst.append(mean(branches_gpt) if branches_gpt else 0)
        acc_lst.append(correct_cnt / total)

    return prob_sat_lst, mean_br_lst, acc_lst





def plot_metric(metric_dict, ylabel, title, outfile, ylim=None):
    """
    绘制不同模型的指标变化曲线，避免 marker 重叠。

    参数:
    - metric_dict : dict，格式为 {model_name: list_of_values}
    - ylabel : str，y 轴标签
    - title : str，图标题
    - outfile : str，保存的图片文件名
    - ylim : tuple(ymin, ymax)，可选
    """
    plt.figure(figsize=(10, 6))

    # 可选 marker 和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'x', '+', '*']
    colors = plt.cm.tab10.colors
    jitter_strength = 0.002  # 控制微调强度

    for i, (model, values) in enumerate(metric_dict.items()):
        # 为每个模型加上轻微抖动，避免重叠
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
        jittered_values = [v + j for v, j in zip(values, jitter)]

        plt.plot(
            ALPHAS, jittered_values,
            marker=markers[i % len(markers)],
            label=model,
            linestyle='-',
            color=colors[i % len(colors)],
            markersize=8,
            markeredgewidth=1,
            markeredgecolor='black'
        )

    plt.xlabel("L / N")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile))
    plt.close()


# ---------- 主流程 ----------
all_prob_sat   = {}
all_mean_br    = {}
all_accuracy   = {}

for mdl in MODELS:
    print(f"→ 统计 {mdl} …")
    prob_sat, mean_br, acc = collect_metrics_for_model(mdl)
    all_prob_sat[mdl] = prob_sat
    all_mean_br[mdl]  = mean_br
    all_accuracy[mdl] = acc

# 多模型对比图
plot_metric(
    all_prob_sat,
    ylabel = "SAT Probability",
    title  = "SAT Probability Comparison Across Models",
    outfile= "compare_sat_prob.png"
)

plot_metric(
    all_mean_br,
    ylabel = "Mean Branches",
    title  = "Mean Branches Comparison Across Models",
    outfile= "compare_mean_branches.png"
)

plot_metric(
    all_accuracy,
    ylabel = "SAT Prediction Accuracy",
    title  = "SAT Prediction Accuracy vs CDCL Across Models",
    outfile= "compare_accuracy.png",
    ylim   = (0, 1.05)
)


