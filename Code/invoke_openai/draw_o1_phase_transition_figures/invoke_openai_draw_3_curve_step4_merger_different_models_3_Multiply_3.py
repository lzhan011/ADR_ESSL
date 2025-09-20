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
MODELS = [ '_CDCL',
           '_openai_prediction_o1',
           '_gpt-5_no_batch',
           '_gpt-3.5-turbo-0125',
           'deepseek-chat',
           'deepseek-reasoner',
           '_claude-3-opus-20240229',
           '_claude-sonnet-4-20250514',
           '_claude-3-7-sonnet-20250219',
           '_claude-3-5-haiku-20241022',
           '_o3-mini',

           ] # 'openai_prediction_o1',

output_dir_base = "draw_o1_cnf_alpha_3_6_N_75"
suffix = ''  # 例如 '_small_alpha'

ALPHAS = np.arange(3.0, 6.0, 0.5)        # L / N

N               = 75
INSTANCES_EACH  = 300
FIG_DIR_ROOT    = 'figures_comparison_1800_instances'
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

def _disp_label(name: str) -> str:
    return name.lstrip('_')  # 去掉前导下划线；也可顺便 .replace('_',' ') 美化

def extract_info_from_file_bak(filepath: str):
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




def extract_info_from_file_method_1(filepath: str, model_select):
    # print("filepath:", filepath)

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
    error_signal = False
    max_line_index = len(lines) -1
    for line_index in range(len(lines)-1, len(lines)-10, -1):
        line = lines[line_index]
        # if line_index == 230:
        #     print(line_index)

        if 'SATISFIABLE OR UNSATISFIABLE:' in line.upper():
            line= line.upper().replace('SATISFIABLE OR UNSATISFIABLE:', '')

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
        else:
            pass


        if 'error' in line.lower():
            error_signal = True

        equal_symbol = re.findall(r'[=]', line)
        if len(equal_symbol) > 3:
            if 'unsatisfiable' in line.lower():
                sat = False
                sat_index = line_index
            elif 'satisfiable' in line.lower():
                sat = True
                sat_index = line_index
            else:
                pass


            if sat is None:
                line_above = lines[line_index - 1]
                if 'unsatisfiable' in line_above.lower():
                    sat = False
                    sat_index = line_index - 1
                elif 'satisfiable' in line_above.lower():
                    sat = True
                    sat_index = line_index - 1
                else:
                    pass



        if 'time' in line and 'seconds' in line:
            # print("line:", line)
            time_val = re.findall(r"\d+\.\d+|\d+", line)[0]

        if 'the formula is satisfiable' in line.lower() and 'UNSATISFIABLE' not in line.upper() and 'if the formula is satisfiable' not in line.lower():
            sat = True
            sat_index = line_index
        elif 'the formula is unsatisfiable' in line.lower():
            sat = False
            sat_index = line_index
        elif 'formula is unsatisfiable' in line.lower():
            sat = False
            sat_index = line_index

        if 'satisfiability' in line.lower() and 'SATISFIABLE' in line.upper() and 'UNSATISFIABLE' not in line.upper():
            sat = True
            sat_index = line_index
        elif 'satisfiability' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index
        if 'the formula is' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index
        if 'formula status' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index



        if sat_index is not  None:
            break


    if 'cnf_k3_N60_L222_alpha3.7_inst3189.txt' in filepath:
        print(filepath)
        pass


    if sat_index == max_line_index or sat_index == max_line_index -1:
        sat_index = None

    if sat_index is not None:

        if 'cnf_k3_N8_L28_alpha3.5_inst1261_RC2_fixed' in filepath and model_select == 'gpt-3.5-turbo-0125':
            sat_index = sat_index - 3

        sat_index_next = sat_index + 1
        # print(filepath)



        sat_index_next = lines[sat_index_next].strip()
        if len(sat_index_next) == 0:
            sat_index = sat_index + 1


        try:
            branches_index = sat_index + 1
            conflicts_index = sat_index + 2

            if "SATISFIABLE" in lines[branches_index].upper():
                branches_index = sat_index + 2
                conflicts_index = sat_index + 3
            if "SATISFIABLE" in lines[sat_index + 2].upper():
                branches_index = sat_index + 3
                conflicts_index = sat_index + 4

            branches_number = lines[branches_index]
            branches_number = re.findall(r'\d+', branches_number)
            branches_number = int(''.join(branches_number))


            conflicts_number = lines[conflicts_index]
            conflicts_number = re.findall(r'\d*', conflicts_number)
            conflicts_number = int(''.join(conflicts_number))
        except:
            print("please check the file:", filepath)

    if sat is not None:
        if branches_number == [] or conflicts_number is None or conflicts_number == "":
            for line_index in range(len(lines) - 1, len(lines) - 10, -1):
                line = lines[line_index].lower()
                if "branches" in line.lower():
                    branches_index = line_index
                    branches_index_next = line_index + 1
                    branches_index_next = min(branches_index_next, len(lines) - 1)
                    for item in [branches_index, branches_index_next]:
                        branches_number = lines[item]
                        branches_number = re.findall(r'\d+', branches_number)
                        if len(branches_number) >= 1:
                            branches_number = int(''.join(branches_number))
                            break
                if "conflicts" in line.lower():
                    conflicts_index = line_index
                    conflicts_index_next = line_index + 1
                    conflicts_index_next = min(len(lines) - 1, conflicts_index_next)
                    for item in [conflicts_index, conflicts_index_next]:
                        conflicts_number = lines[item]
                        conflicts_number = re.findall(r'\d+', conflicts_number)
                        if len(conflicts_number) >= 1:
                            conflicts_number = int(''.join(conflicts_number))
                            break





    return time_val, branches_number, conflicts_number, sat, clauses, error_signal



def extract_info_from_file(filepath: str, model_select):
    # print("filepath:", filepath)

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
    error_signal = False
    max_line_index = len(lines) -1
    for line_index in range(len(lines)):
        line = lines[line_index]
        # if line_index == 230:
        #     print(line_index)
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
        else:
            pass


        if 'error' in line.lower():
            error_signal = True

        equal_symbol = re.findall(r'[=]', line)
        if len(equal_symbol) > 3:
            if 'unsatisfiable' in line.lower():
                sat = False
                sat_index = line_index
            elif 'satisfiable' in line.lower():
                sat = True
                sat_index = line_index
            else:
                pass


            if sat is None:
                line_above = lines[line_index - 1]
                if 'unsatisfiable' in line_above.lower():
                    sat = False
                    sat_index = line_index - 1
                elif 'satisfiable' in line_above.lower():
                    sat = True
                    sat_index = line_index - 1
                else:
                    pass



        if 'time' in line and 'seconds' in line:
            # print("line:", line)
            time_val = re.findall(r"\d+\.\d+|\d+", line)[0]




        if 'the formula is satisfiable' in line.lower() and 'UNSATISFIABLE' not in line.upper() and 'if the formula is satisfiable' not in line.lower():
            sat = True
            sat_index = line_index
        elif 'the formula is unsatisfiable' in line.lower():
            sat = False
            sat_index = line_index
        elif 'formula is unsatisfiable' in line.lower():
            sat = False
            sat_index = line_index

        if 'satisfiability' in line.lower() and 'SATISFIABLE' in line.upper() and 'UNSATISFIABLE' not in line.upper():
            sat = True
            sat_index = line_index
        elif 'satisfiability' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index
        if 'the formula is' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index
        if 'formula status' in line.lower() and 'UNSATISFIABLE' in line.upper():
            sat = False
            sat_index = line_index

        if sat_index is not  None:
            break


    if 'cnf_k3_N60_L222_alpha3.7_inst3189.txt' in filepath:
        print(filepath)
        pass


    if sat_index == max_line_index or sat_index == max_line_index -1:
        sat_index = None

    if sat_index is not None:

        if 'cnf_k3_N8_L28_alpha3.5_inst1261_RC2_fixed' in filepath and model_select == 'gpt-3.5-turbo-0125':
            sat_index = sat_index - 3

        sat_index_next = sat_index + 1
        # print(filepath)



        sat_index_next = lines[sat_index_next].strip()
        if len(sat_index_next) == 0:
            sat_index = sat_index + 1


        try:
            branches_index = sat_index + 1
            conflicts_index = sat_index + 2

            if "SATISFIABLE" in lines[branches_index].upper():
                branches_index = sat_index + 2
                conflicts_index = sat_index + 3
            if "SATISFIABLE" in lines[sat_index + 2].upper():
                branches_index = sat_index + 3
                conflicts_index = sat_index + 4

            branches_number = lines[branches_index]
            branches_number = re.findall(r'\d+', branches_number)
            branches_number = int(''.join(branches_number))


            conflicts_number = lines[conflicts_index]
            conflicts_number = re.findall(r'\d*', conflicts_number)
            conflicts_number = int(''.join(conflicts_number))
        except:
            print("please check the file:", filepath)




    return time_val, branches_number, conflicts_number, sat, clauses, error_signal




def _safe_div(a, b):
    return float(a) / float(b) if b else np.nan

def _f1_from_pr(precision, recall):
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return np.nan
    return 2 * precision * recall / (precision + recall)

def set_output_dir(output_dir_base, model, suffix):
    if model.startswith('deepseek-reasoner'):
        output_dir_base_deepseek = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_deepseek/draw_deepseek_phase_transition_figures/'
        output_dir = "draw_deepseek_cnf_alpha_3_6_N_75_openai_prediction_" + model + suffix
        output_dir = os.path.join(output_dir_base_deepseek,  output_dir)
    elif model.startswith('anthropic'):
        output_dir_base_deepseek = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_anthropic'
        output_dir = output_dir_base + model + suffix
        output_dir = os.path.join(output_dir_base_deepseek, output_dir)
    else:
        output_dir = output_dir_base + model + suffix
    return output_dir


from typing import Tuple

# 单个 findall，把 N/L/alpha/inst 一次性抓出来（按固定顺序）
_PATTERN = r'(?i)N(\d+).*?L(\d+).*?alpha(\d+(?:\.\d+)?).*?inst(\d+)'

def parse_cnf_fname(fname: str) -> Tuple[int, float, int]:
    """
    解析类似：
      k3_N75_L337_alpha4.5_inst49_UNSAT.cnf
      cnf_k3_N75_L300_alpha4.0_inst12.txt
    等文件名，返回 (N:int, alpha:float, idx:int)
    """
    base = os.path.basename(fname)
    hits = re.findall(_PATTERN, base)
    if not hits:
        raise ValueError(f"Unrecognized filename format: {base}")
    N_str, L_str, alpha_str, idx_str = hits[0]
    return int(N_str), float(alpha_str), int(idx_str)

# 如果你也想拿到 L，就用这个变体：
def parse_cnf_fname_with_L(fname: str) -> Tuple[int, float, int, int]:
    base = os.path.basename(fname)
    hits = re.findall(_PATTERN, base)
    if not hits:
        raise ValueError(f"Unrecognized filename format: {base}")
    N_str, L_str, alpha_str, idx_str = hits[0]
    return int(N_str), float(alpha_str), int(idx_str), int(L_str)

# ---------- 指标收集 ----------
def collect_metrics_for_model(model: str):
    """
    返回（均为按 alpha 排列的列表）：
      prob_sat_list, median_branches_list, acc_list,
      prec_sat_list, rec_sat_list, f1_sat_list,
      prec_unsat_list, rec_unsat_list, f1_unsat_list,   # NEW
      prf_records（逐 alpha 行）
    """
    output_dir = set_output_dir(output_dir_base, model, suffix)
    prob_sat_lst, median_br_lst, acc_lst = [], [], []
    prec_sat_list, rec_sat_list, f1_sat_list = [], [], []
    prec_unsat_list, rec_unsat_list, f1_unsat_list = [], [], []  # NEW
    prf_records = []
    file_records = []  # ← 新增
    error_signal_cnt = 0
    for alpha in ALPHAS:
        branches_gpt = []
        tp = fp = fn = tn = 0
        sat_gpt_cnt = 0
        correct_cnt, total_cnt = 0, 0

        for fname in os.listdir(output_dir):
            N, alpha_f, idx, L =  parse_cnf_fname_with_L(fname)
            if alpha_f != alpha:
                continue
            path  = os.path.join(output_dir, fname)
            if not os.path.exists(path):
                continue

            if "k3_N75_L262_alpha3.5_inst137_SAT.cnf" in path:
                print("path:", path)

            _, br, _, gpt_sat, clauses, error_signal = extract_info_from_file(path, model)
            _, br_m1, _, gpt_sat_m1, _, error_signal_m1 = extract_info_from_file_method_1(path, model)

            if gpt_sat is not None or br is not None:
                br = br_m1
                gpt_sat = gpt_sat_m1

            if error_signal:
                error_signal_cnt += 1

            if gpt_sat is None or br is None:
                continue

            if isinstance(br, list) :
                print("list", fname)

            if  br >= 1000:
                print("br>100", fname)
                continue

            total_cnt += 1
            branches_gpt.append(br)
            if gpt_sat:
                sat_gpt_cnt += 1

            with Minisat22(bootstrap_with=clauses) as m:
                mini_sat = m.solve()

            # ← 新增：逐文件记录一行
            file_records.append({
                "model": model,
                "filename": fname,
                "alpha": float(alpha_f),   # 或 float(alpha)
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

        # 以 UNSAT 为正类（等价于把负类当正类）
        # tp_unsat = tn, fp_unsat = fn, fn_unsat = fp, tn_unsat = tp
        precision_unsat = _safe_div(tn, (tn + fn))
        recall_unsat    = _safe_div(tn, (tn + fp))
        f1_unsat        = _f1_from_pr(precision_unsat, recall_unsat)

        # 收集
        prob_sat_lst.append(prob_sat)
        median_br_lst.append(med_br)
        acc_lst.append(acc)

        prec_sat_list.append(precision_sat)
        rec_sat_list.append(recall_sat)
        f1_sat_list.append(f1_sat)

        prec_unsat_list.append(precision_unsat)   # NEW
        rec_unsat_list.append(recall_unsat)       # NEW
        f1_unsat_list.append(f1_unsat)            # NEW

        prf_records.append({
            "model": model, "alpha": float(alpha), "n": total_cnt,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc,
            "precision_sat": precision_sat, "recall_sat": recall_sat, "f1_sat": f1_sat,            # UPDATED
            "precision_unsat": precision_unsat, "recall_unsat": recall_unsat, "f1_unsat": f1_unsat, # NEW
            "sat_prob_pred": prob_sat, "median_branches_pred": med_br
        })

    print("error_signal_cnt:", error_signal_cnt)
    return (prob_sat_lst, median_br_lst, acc_lst,
            prec_sat_list, rec_sat_list, f1_sat_list,
            prec_unsat_list, rec_unsat_list, f1_unsat_list, prf_records, file_records)  # UPDATED

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
            label=_disp_label(m), linestyle='-', markeredgewidth=1.5, markeredgecolor='black'
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
                label=_disp_label(m), linestyle='-', markeredgewidth=1.5, markeredgecolor='black'
            )
        ax.set_xlabel("L / N")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, linewidth=1)

    handles = [Line2D([0], [0], color=_MODEL_COLOR[m], marker=_MODEL_MARKER[m],
                      linestyle='-', markeredgewidth=1.5, markeredgecolor='black', label=_disp_label(m))
               for m in MODELS]
    labels = [_disp_label(m) for m in MODELS]
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
                label=_disp_label(m), linestyle='-', markeredgewidth=1.5, markeredgecolor='black'
            )
        ax.set_xlabel("L / N")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, linewidth=1)

    handles = [Line2D([0], [0], color=_MODEL_COLOR[m], marker=_MODEL_MARKER[m],
                      linestyle='-', markeredgewidth=1.5, markeredgecolor='black', label=_disp_label(m))
               for m in MODELS]
    labels = [_disp_label(m) for m in MODELS]
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
rows_per_file = []   # ← 新增
for mdl in MODELS:
    print(f"→ 统计 {mdl} …")
    (prob_sat, med_br, acc,
     prec_sat, rec_sat, f1_sat,
     prec_unsat, rec_unsat, f1_unsat, prf, file_recs) = collect_metrics_for_model(mdl)  # UPDATED

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
    rows_per_file.extend(file_recs)  # ← 新增

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
csv_path = os.path.join(FIG_DIR, "per_file_predictions.xlsx")
df_files.to_excel(csv_path, index=False)
print(f"[OK] 逐文件记录已保存: {csv_path}")


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
