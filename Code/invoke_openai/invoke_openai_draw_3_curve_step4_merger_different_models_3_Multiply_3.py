"""
multi_model_compare.py  (updated)
======================

比较多个大模型在随机 3-SAT 上的表现：
  • SAT 概率 (LLM 预测)
  • 中位分支数 (LLM 报告的 branches 的 median)
  • 准确率 Accuracy（与 MiniSat 真值比较）
  • Precision / Recall / F1（以 SAT 为正类）
  • Precision / Recall / F1（以 UNSAT 为正类）  # NEW
并导出 Excel；绘制单图 + 6-宫格大图 + 3×3 大图（统一图例在下方）。  # UPDATED
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
# MODELS = ['gpt-4o_batch', 'o1_input_based__gpt-4o',] # 'gpt-5_batch_reasoning_low',
MODELS = ['o1_input_based_CDCL',
          'o1_input_based__gpt-3.5-turbo-0125',
          'o1_input_based__chatgpt-4o-latest',
          'o1_input_based_gpt-4.1',
          'o1',
          'o1_input_based_gpt-5_no_batch',
          'gpt-5_batch_reasoning_low',
          'o1_input_based__chatgpt-4o-latest',
          'o1_input_based_deepseek-chat',
          'o1_input_based_deepseek-reasoner',
          'o1_input_based__claude-3-7-sonnet-20250219',
          'o1_input_based__claude-3-opus-20240229',
          'o1_input_based__claude-sonnet-4-20250514',
          'o1_input_based__claude-3-5-haiku-20241022',

          ]

output_dir_base = "cnf_results_openai_"
suffix = ''  # 例如 '_small_alpha'

ALPHAS = np.arange(3.0, 6.0, 0.5)        # L / N

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


def collect_metrics_for_model(model: str):
    """
    返回（均为按 alpha 排列的列表）：
      prob_sat_list, median_branches_list, acc_list,
      prec_sat_list, rec_sat_list, f1_sat_list,
      prec_unsat_list, rec_unsat_list, f1_unsat_list,
      prf_records（逐 alpha 行）,
      file_records（逐文件行，新增）
    """
    output_dir = set_output_dir(output_dir_base, model, suffix)
    prob_sat_lst, median_br_lst, acc_lst = [], [], []
    prec_sat_list, rec_sat_list, f1_sat_list = [], [], []
    prec_unsat_list, rec_unsat_list, f1_unsat_list = [], [], []
    prf_records = []
    file_records = []   # ← NEW

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

            # —— 逐文件记录（只要能得到 gpt_sat 与 mini_sat 就写一行）
            file_records.append({
                "model": model,
                "filename": fname,
                "alpha": float(alpha),
                "N": int(N),
                "ground_truth_sat": bool(mini_sat),
                "predicted_sat": bool(gpt_sat),
                "branches_number":br,
            })

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

        # 以 SAT 为正类
        precision_sat = _safe_div(tp, (tp + fp))
        recall_sat    = _safe_div(tp, (tp + fn))
        f1_sat        = _f1_from_pr(precision_sat, recall_sat)

        # 以 UNSAT 为正类
        precision_unsat = _safe_div(tn, (tn + fn))
        recall_unsat    = _safe_div(tn, (tn + fp))
        f1_unsat        = _f1_from_pr(precision_unsat, recall_unsat)

        prob_sat_lst.append(prob_sat)
        median_br_lst.append(med_br)
        acc_lst.append(acc)
        prec_sat_list.append(precision_sat)
        rec_sat_list.append(recall_sat)
        f1_sat_list.append(f1_sat)
        prec_unsat_list.append(precision_unsat)
        rec_unsat_list.append(recall_unsat)
        f1_unsat_list.append(f1_unsat)

        prf_records.append({
            "model": model, "alpha": float(alpha), "n": total_cnt,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc,
            "precision_sat": precision_sat, "recall_sat": recall_sat, "f1_sat": f1_sat,
            "precision_unsat": precision_unsat, "recall_unsat": recall_unsat, "f1_unsat": f1_unsat,
            "sat_prob_pred": prob_sat, "median_branches_pred": med_br
        })

    return (prob_sat_lst, median_br_lst, acc_lst,
            prec_sat_list, rec_sat_list, f1_sat_list,
            prec_unsat_list, rec_unsat_list, f1_unsat_list,
            prf_records, file_records)  # ← 在末尾多返回 file_records


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
        if i >= 220:
            print(i)
    # for i in range(len(lines)-1, -1, -1):
        line = lines[i]
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

# ---------- 画图：单图（保留） ----------
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
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------- 画图：6-宫格（保留） ----------
def plot_six_panel(all_sat_prob, all_med_br, all_accuracy, all_precision_sat, all_recall_sat, all_f1_sat,
                   outfile_base="compare_6panels"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    metric_list = [
        ("Median Branches", all_med_br,         "(a) Median branches"),
        ("SAT Probability", all_sat_prob,       "(b) SAT probability"),
        ("Accuracy",        all_accuracy,       "(c) Accuracy"),
        ("Precision(SAT+)", all_precision_sat,  "(d) Precision (SAT positive)"),
        ("Recall(SAT+)",    all_recall_sat,     "(e) Recall (SAT positive)"),
        ("F1(SAT+)",        all_f1_sat,         "(f) F1 (SAT positive)"),
    ]
    ylims = [None, (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05)]

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

    handles = [Line2D([0], [0], color=_MODEL_COLOR[m], marker=_MODEL_MARKER[m],
                      linestyle='-', markeredgewidth=1.5, markeredgecolor='black', label=m)
               for m in MODELS]
    labels = MODELS
    fig.legend(handles, labels, loc='lower center', ncol=2,
               frameon=True, bbox_to_anchor=(0.5, -0.20))
    plt.subplots_adjust(wspace=0.28, hspace=0.35, bottom=0.12)
    for ext in ("png","pdf","svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{outfile_base}.{ext}"), dpi=300 if ext=="png" else None, bbox_inches='tight')
    plt.close(fig)

# ---------- 画图：9-宫格（3×3，新增） ----------  # NEW
def plot_nine_panel(all_sat_prob, all_med_br, all_accuracy,
                    all_precision_sat, all_recall_sat, all_f1_sat,
                    all_precision_unsat, all_recall_unsat, all_f1_unsat,
                    outfile_base="compare_all_metrics_3x3"):
    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    axes = axes.ravel()
    metric_list = [
        ("Median Branches",  all_med_br,          "(a) Median branches"),
        ("SAT Probability",  all_sat_prob,        "(b) SAT probability"),
        ("Accuracy",         all_accuracy,        "(c) Accuracy"),
        ("Precision(SAT+)",  all_precision_sat,   "(d) Precision (SAT positive)"),
        ("Recall(SAT+)",     all_recall_sat,      "(e) Recall (SAT positive)"),
        ("F1(SAT+)",         all_f1_sat,          "(f) F1 (SAT positive)"),
        ("Precision(UNSAT+)",all_precision_unsat, "(g) Precision (UNSAT positive)"),
        ("Recall(UNSAT+)",   all_recall_unsat,    "(h) Recall (UNSAT positive)"),
        ("F1(UNSAT+)",       all_f1_unsat,        "(i) F1 (UNSAT positive)"),
    ]
    ylims = [None, (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05), (0,1.05)]

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

    handles = [Line2D([0], [0], color=_MODEL_COLOR[m], marker=_MODEL_MARKER[m],
                      linestyle='-', markeredgewidth=1.5, markeredgecolor='black', label=m)
               for m in MODELS]
    labels = MODELS
    fig.legend(handles, labels, loc='lower center', ncol=min(2, len(MODELS)),
               frameon=True, bbox_to_anchor=(0.5, -0.06))
    plt.subplots_adjust(wspace=0.30, hspace=0.40, bottom=0.12)
    for ext in ("png","pdf","svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{outfile_base}.{ext}"), dpi=300 if ext=="png" else None, bbox_inches='tight')
    plt.close(fig)

# ---------- 主流程 ----------
all_sat_prob   = {}
all_med_br     = {}
all_accuracy   = {}
all_precision_sat  = {}
all_recall_sat     = {}
all_f1_sat         = {}
all_precision_unsat= {}   # NEW
all_recall_unsat   = {}   # NEW
all_f1_unsat       = {}   # NEW
rows = []
rows_per_file = []   # ← NEW

for mdl in MODELS:
    print(f"→ 统计 {mdl} …")
    (prob_sat, med_br, acc,
     prec_sat, rec_sat, f1_sat,
     prec_unsat, rec_unsat, f1_unsat,
     prf, file_recs) = collect_metrics_for_model(mdl)  # ← UPDATED

    all_sat_prob[mdl]        = prob_sat
    all_med_br[mdl]          = med_br
    all_accuracy[mdl]        = acc
    all_precision_sat[mdl]   = prec_sat
    all_recall_sat[mdl]      = rec_sat
    all_f1_sat[mdl]          = f1_sat
    all_precision_unsat[mdl] = prec_unsat    # NEW
    all_recall_unsat[mdl]    = rec_unsat     # NEW
    all_f1_unsat[mdl]        = f1_unsat      # NEW
    rows.extend(prf)
    rows_per_file.extend(file_recs)  # ← NEW

# 导出 Excel（模型×α）
df = pd.DataFrame(rows, columns=[
    "model","alpha","n","tp","fp","fn","tn",
    "accuracy",
    "precision_sat","recall_sat","f1_sat",
    "precision_unsat","recall_unsat","f1_unsat",
    "sat_prob_pred","median_branches_pred"
])
excel_path = os.path.join(FIG_DIR, "metrics_by_model_alpha.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="metrics")
print(f"[OK] 指标表已保存: {excel_path}")

# === 逐文件记录保存 ===
df_files = pd.DataFrame(rows_per_file, columns=[
    "model", "filename", "alpha", "N", "ground_truth_sat", "predicted_sat", "branches_number"
])
xlsx_path = os.path.join(FIG_DIR, "per_file_predictions.xlsx")
df_files.to_excel(xlsx_path, index=False)
print(f"[OK] 逐文件记录已保存: {xlsx_path}")

# 单图（可选）
plot_metric(all_sat_prob, ylabel="SAT Probability",
            title="SAT Probability Comparison Across Models",
            outfile="compare_sat_prob.png", ylim=(0,1.05))
plot_metric(all_med_br, ylabel="Median Branches",
            title="Median Branches Comparison Across Models",
            outfile="compare_median_branches.png")
plot_metric(all_accuracy, ylabel="Accuracy",
            title="SAT Prediction Accuracy vs CDCL Across Models",
            outfile="compare_accuracy.png", ylim=(0,1.05))
plot_metric(all_precision_sat, ylabel="Precision (SAT positive)",
            title="SAT Precision (SAT as Positive)",
            outfile="compare_precision_sat.png", ylim=(0,1.05))
plot_metric(all_recall_sat, ylabel="Recall (SAT positive)",
            title="SAT Recall (SAT as Positive)",
            outfile="compare_recall_sat.png", ylim=(0,1.05))
plot_metric(all_f1_sat, ylabel="F1 (SAT positive)",
            title="SAT F1 (SAT as Positive)",
            outfile="compare_f1_sat.png", ylim=(0,1.05))
# 新增：UNSAT 为正类的三张单图
plot_metric(all_precision_unsat, ylabel="Precision (UNSAT positive)",
            title="UNSAT Precision (UNSAT as Positive)",
            outfile="compare_precision_unsat.png", ylim=(0,1.05))
plot_metric(all_recall_unsat, ylabel="Recall (UNSAT positive)",
            title="UNSAT Recall (UNSAT as Positive)",
            outfile="compare_recall_unsat.png", ylim=(0,1.05))
plot_metric(all_f1_unsat, ylabel="F1 (UNSAT positive)",
            title="UNSAT F1 (UNSAT as Positive)",
            outfile="compare_f1_unsat.png", ylim=(0,1.05))

# 6-宫格：与原来一致（SAT 为正类）
plot_six_panel(all_sat_prob, all_med_br, all_accuracy,
               all_precision_sat, all_recall_sat, all_f1_sat,
               outfile_base="compare_all_metrics_2x3")

# 3×3 九宫格：加入“分类”的三项（UNSAT 为正类）
plot_nine_panel(all_sat_prob, all_med_br, all_accuracy,
                all_precision_sat, all_recall_sat, all_f1_sat,
                all_precision_unsat, all_recall_unsat, all_f1_unsat,
                outfile_base="compare_all_metrics_3x3")

print(f"[OK] 图片输出目录: {FIG_DIR}")
