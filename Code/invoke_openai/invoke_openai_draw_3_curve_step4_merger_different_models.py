"""
multi_model_compare.py
======================

比较多个大模型在随机 3-SAT 上的表现：
  • SAT 概率 (LLM 预测)
  • 中位分支数 (LLM 报告的 branches 的 median)
  • 准确率 Accuracy（与 MiniSat 真值比较）
  • Precision / Recall / F1（以 SAT 为正类）
并导出 Excel；绘制单图 + 6-宫格大图（统一图例在下方）。
"""

import os
import re
from statistics import median
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pysat.solvers import Minisat22
import pandas as pd

# ---------- 配置 ----------
MODELS = ['gpt-3.5-turbo', 'gpt-4o',  'gpt-4.1', 'o1']   # 固定顺序，保证图例一致
MODELS =  ['gpt-4o_batch', 'o1',
           'gpt-5',
           'gpt-5_batch_reasoning_low',
           'o1_input_based__gpt-4o',
           'o1_input_based_gpt-4.1',
           'o1_input_based__gpt-4-turbo',
           'o1_input_based__chatgpt-4o-latest',
           'o1_input_based_deepseek-chat',
           'o1_input_based__gpt-3.5-turbo',
           'o1_input_based__gpt-3.5-turbo-0125']
MODELS = ['gpt-4o_batch', 'o1_input_based__gpt-4o',]

output_dir_base = "cnf_results_openai_"
suffix = ''  # 例如 '_small_alpha'

ALPHAS = np.arange(3.0, 6.0, 0.5)        # L / N
# 小 alpha 场景可改为：
# ALPHAS = np.arange(1.0, 4.5, 0.5)

N               = 75
INSTANCES_EACH  = 20
FIG_DIR_ROOT    = 'figures_comparison'
FIG_DIR         = os.path.join(FIG_DIR_ROOT, output_dir_base + suffix)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 画图样式（论文友好） ----------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
})
_TAB10 = plt.cm.tab10.colors
_MODEL_COLOR = {m: _TAB10[i % len(_TAB10)] for i, m in enumerate(MODELS)}
_MARKERS = ['o', 's', '^', 'D', 'v']
_MODEL_MARKER = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(MODELS)}

# ---------- 解析工具 ----------
def _safe_int_from_line(s: str) -> int:
    nums = re.findall(r'\d+', s)
    return int(nums[0]) if nums else 0

def extract_info_from_file(filepath: str):
    """
    返回:
      time_val(float,未用), branches(int or None), conflicts(int or None),
      gpt_sat(bool or None), clauses(list[list[int]])
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses, read_clause = [], False
    time_val, branches_number, conflicts_number = 0.0, None, None
    sat = None
    sat_index = None

    for i, line in enumerate(lines):
        if line.startswith("p cnf"):
            read_clause = True
            continue
        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
            else:
                clause = [int(x) for x in line.strip().split() if x != "0"]
                if clause:
                    clauses.append(clause)
            continue

        letters_word = ''.join(re.findall(r'[a-zA-Z]', line)).lower()
        if letters_word == "unsatisfiable":
            sat = False
            sat_index = i
        elif letters_word == "satisfiable":
            sat = True
            sat_index = i

    if sat_index is not None:
        if sat_index + 1 < len(lines):
            branches_number = _safe_int_from_line(lines[sat_index + 1])
        if sat_index + 2 < len(lines):
            conflicts_number = _safe_int_from_line(lines[sat_index + 2])

    return time_val, branches_number, conflicts_number, sat, clauses

def _safe_div(a, b):
    return float(a) / float(b) if b else np.nan

def _f1_from_pr(precision, recall):
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return np.nan
    return 2 * precision * recall / (precision + recall)



def set_output_dir(output_dir_base, model, suffix):
    if model.startswith('deepseek'):
        output_dir_base_deepseek = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_deepseek'
        output_dir = output_dir_base + model + suffix
        output_dir = os.path.join(output_dir_base_deepseek,  output_dir)
    elif model.startswith('anthropic'):
        output_dir_base_deepseek = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_anthropic'
        output_dir = output_dir_base + model + suffix
        output_dir = os.path.join(output_dir_base_deepseek, output_dir)
    else:
        output_dir = output_dir_base + model + suffix


    return output_dir



# ---------- 指标收集 ----------
def collect_metrics_for_model(model: str):
    """
    返回：
      prob_sat_list, median_branches_list, acc_list,
      prec_list, rec_list, f1_list,
      prf_records (用于汇总表)
    """
    # output_dir = output_dir_base + model + suffix
    output_dir = set_output_dir(output_dir_base, model, suffix)
    prob_sat_lst, median_br_lst, acc_lst = [], [], []
    prec_list, rec_list, f1_list = [], [], []
    prf_records = []

    for alpha in ALPHAS:
        branches_gpt = []
        tp = fp = fn = tn = 0
        sat_gpt_cnt = 0
        correct_cnt, total_cnt = 0, 0

        for idx in range(1, INSTANCES_EACH + 1):
            fname = f"cnf_k3_N{N}_L{int(alpha*N)}_alpha{alpha:.1f}_inst{idx}.txt"
            path  = os.path.join(output_dir, fname)
            if not os.path.exists(path):
                continue

            _, br, _, gpt_sat, clauses = extract_info_from_file(path)
            if gpt_sat is None or br is None:
                continue

            total_cnt += 1
            branches_gpt.append(br)
            if gpt_sat:
                sat_gpt_cnt += 1

            with Minisat22(bootstrap_with=clauses) as m:
                mini_sat = m.solve()

            if mini_sat == gpt_sat:
                correct_cnt += 1

            # 混淆矩阵（正类=SAT）
            if mini_sat and gpt_sat:
                tp += 1
            elif (not mini_sat) and gpt_sat:
                fp += 1
            elif mini_sat and (not gpt_sat):
                fn += 1
            else:
                tn += 1

        total = total_cnt if total_cnt > 0 else 1
        prob_sat = sat_gpt_cnt / total
        med_br   = median(branches_gpt) if branches_gpt else 0.0
        acc      = correct_cnt / total
        precision = _safe_div(tp, (tp + fp))
        recall    = _safe_div(tp, (tp + fn))
        f1        = _f1_from_pr(precision, recall)

        prob_sat_lst.append(prob_sat)
        median_br_lst.append(med_br)
        acc_lst.append(acc)
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)

        prf_records.append({
            "model": model, "alpha": float(alpha), "n": total_cnt,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "sat_prob_pred": prob_sat, "median_branches_pred": med_br
        })

    return prob_sat_lst, median_br_lst, acc_lst, prec_list, rec_list, f1_list, prf_records

# ---------- 画图：单图（图例放图外右侧；颜色/marker固定） ----------
def plot_metric(metric_dict, ylabel, title, outfile, ylim=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    jitter_strength = 0.002
    for m in MODELS:
        if m not in metric_dict:
            continue
        values = metric_dict[m]
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
        jittered_values = [v + j for v, j in zip(values, jitter)]
        ax.plot(
            ALPHAS, jittered_values,
            marker=_MODEL_MARKER[m], color=_MODEL_COLOR[m],
            label=m, linestyle='-', markeredgewidth=1.5, markeredgecolor='black'
        )
    ax.set_xlabel("L / N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, linewidth=1)

    # 图外右侧统一位置的图例
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------- 画图：6-宫格大图（统一图例在下方，只显示一次） ----------
def plot_six_panel(all_sat_prob, all_med_br, all_accuracy, all_precision, all_recall, all_f1,
                   outfile_base="compare_6panels"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    metric_list = [
        ("Median Branches", all_med_br,      "(a) Median branches"),
        ("SAT Probability", all_sat_prob,    "(b) SAT probability"),
        ("Accuracy",        all_accuracy,    "(c) Accuracy"),
        ("Precision",       all_precision,   "(d) Precision"),
        ("Recall",          all_recall,      "(e) Recall"),
        ("F1",              all_f1,          "(f) F1"),
    ]
    ylims = [None, (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05)]

    # 逐子图绘制（不放各自图例）
    for ax, (ylabel, data_dict, subtitle), ylim in zip(axes, metric_list, ylims):
        jitter_strength = 0.002
        for m in MODELS:
            if m not in data_dict:
                continue
            values = data_dict[m]
            jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
            jittered_values = [v + j for v, j in zip(values, jitter)]
            ax.plot(
                ALPHAS, jittered_values,
                marker=_MODEL_MARKER[m], color=_MODEL_COLOR[m],
                label=m, linestyle='-', markeredgewidth=1.5, markeredgecolor='black'
            )
        ax.set_xlabel("L / N")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, linewidth=1)

    # 统一图例：用固定的 handles/labels，放在整张图下方
    handles = [Line2D([0], [0], color=_MODEL_COLOR[m], marker=_MODEL_MARKER[m],
                      linestyle='-', markeredgewidth=1.5, markeredgecolor='black', label=m)
               for m in MODELS]
    labels = MODELS
    fig.legend(handles, labels, loc='lower center', ncol=2,
               frameon=True, bbox_to_anchor=(0.5, -0.20))

    plt.subplots_adjust(wspace=0.28, hspace=0.35, bottom=0.12)
    png = os.path.join(FIG_DIR, outfile_base + ".png")
    pdf = os.path.join(FIG_DIR, outfile_base + ".pdf")
    svg = os.path.join(FIG_DIR, outfile_base + ".svg")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)

# ---------- 主流程 ----------
all_sat_prob   = {}
all_med_br     = {}
all_accuracy   = {}
all_precision  = {}
all_recall     = {}
all_f1         = {}
rows = []

for mdl in MODELS:
    print(f"→ 统计 {mdl} …")
    (prob_sat, med_br, acc, prec, rec, f1, prf) = collect_metrics_for_model(mdl)
    all_sat_prob[mdl]  = prob_sat
    all_med_br[mdl]    = med_br
    all_accuracy[mdl]  = acc
    all_precision[mdl] = prec
    all_recall[mdl]    = rec
    all_f1[mdl]        = f1
    rows.extend(prf)

# 导出 Excel（模型×α）
df = pd.DataFrame(rows, columns=[
    "model","alpha","n","tp","fp","fn","tn",
    "accuracy","precision","recall","f1",
    "sat_prob_pred","median_branches_pred"
])
excel_path = os.path.join(FIG_DIR, "metrics_by_model_alpha.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="metrics")
print(f"[OK] 指标表已保存: {excel_path}")

# 单图：图例放外侧右边
plot_metric(all_sat_prob, ylabel="SAT Probability",
            title="SAT Probability Comparison Across Models",
            outfile="compare_sat_prob.png", ylim=(0,1.05))
plot_metric(all_med_br, ylabel="Median Branches",
            title="Median Branches Comparison Across Models",
            outfile="compare_median_branches.png")
plot_metric(all_accuracy, ylabel="Accuracy",
            title="SAT Prediction Accuracy vs CDCL Across Models",
            outfile="compare_accuracy.png", ylim=(0,1.05))
plot_metric(all_precision, ylabel="Precision",
            title="SAT Prediction Precision (Positive = SAT)",
            outfile="compare_precision.png", ylim=(0,1.05))
plot_metric(all_recall, ylabel="Recall",
            title="SAT Prediction Recall (Positive = SAT)",
            outfile="compare_recall.png", ylim=(0,1.05))
plot_metric(all_f1, ylabel="F1",
            title="SAT Prediction F1 (Positive = SAT)",
            outfile="compare_f1.png", ylim=(0,1.05))

# 6-宫格大图：统一图例位于第二排下方
plot_six_panel(all_sat_prob, all_med_br, all_accuracy,
               all_precision, all_recall, all_f1,
               outfile_base="compare_all_metrics_2x3")

print(f"[OK] 图片输出目录: {FIG_DIR}")
