# 创建一个 Python 脚本文件，内容为使用 Minisat22 解析 CNF 并绘图的代码

import os
import re
import numpy as np
from statistics import mean, median
import matplotlib.pyplot as plt
from pysat.solvers import Minisat22

# 参数设定
input_dir = "draw_o1_cnf_alpha_3_6_N_75_openai_prediction_"
alpha_values = np.arange(3.0, 6.0, 0.5)


model_selected = 'o1' # 'gpt-4-turbo' # 'chatgpt-4o-latest'  #'gpt-4.1' # 'gpt-4o' #'gpt-3.5-turbo'  'gpt-3.5-turbo-0125'    'o1'
# o3,  o4-mini,

# o1-pro, o3, o3-mini, o3-pro, o3-deep-research, o4-mini,
# 输出目录
# output_dir = "cnf_results_openai_"+model_selected + "_small_alpha"
input_dir = input_dir+model_selected
# output_dir = "cnf_results_openai_gpt-4o_small_alpha"
# output_dir = "cnf_results_openai_gpt-3.5-turbo_small_alpha"
# alpha_values = np.arange(1.0, 4.5, 0.5)
alpha_values = np.arange(3.0, 6.0, 0.5)

output_dir_figure = os.path.join('figures', input_dir, "GPT_correct_flag_False")
os.makedirs('figures', exist_ok=True)
os.makedirs(output_dir_figure, exist_ok=True)

N = 75
k = 3
instances_per_alpha = 300
gpt_correct_flag_selected = True
# 初始化容器
mean_branches_gpt, median_branches_gpt, prob_sat_gpt, avg_times_gpt = [], [], [], []
mean_branches_mini, median_branches_mini, prob_sat_mini = [], [], []
gpt_vs_mini_accuracy = []


# ---------- 工具函数 ----------
def extract_info_from_file(filepath: str):
    # print("filepath:", filepath)

    """
    解析单个结果文件，返回
      time, branches, conflicts, gpt_sat(bool/None), clauses(list[list[int]])
    —— 你原来的 extract_info_from_file() 函数直接拿来即可。
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [item.strip() for item in lines]
    clauses, read_clause = [], False
    time_val, branches, conflicts = 0.0, 0, 0
    sat = None    # True / False / None
    sat_index = None
    branches_number, conflicts_number = None, None
    for line_index in range(len(lines)):

        # if line_index == 320:
        #     print(lines[line_index])

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
        sat_index_next = sat_index + 1
        sat_index_next = lines[sat_index_next].strip()
        if len(sat_index_next) == 0:
            sat_index = sat_index + 1

        try:
            branches_index = sat_index + 1
            branches_number = lines[branches_index].strip()
            branches_number = re.findall(r'\d*', branches_number)
            branches_number = int(''.join(branches_number))

            conflicts_index = sat_index + 2
            conflicts_number = lines[conflicts_index]
            conflicts_number = re.findall(r'\d*', conflicts_number)
            conflicts_number = int(''.join(conflicts_number))
        except:
            print("please check the file:", filepath)




    return time_val, branches_number, conflicts_number, sat, clauses




def set_value_for_specific_files(file_path, gpt_sat, branches_number, conflicts_number):
    if "k3_N75_L262_alpha3.5_inst137_SAT.cnf" in file_path:
        gpt_sat = False
        branches_number = 94
        conflicts_number = 30
    if "k3_N75_L262_alpha3.5_inst109_SAT" in file_path:
        gpt_sat = False
        branches_number = 44
        conflicts_number = 29
    if "k3_N75_L300_alpha4.0_inst270_SAT" in file_path:
        gpt_sat = False
        branches_number = 37
        conflicts_number = 19

    return  gpt_sat, branches_number, conflicts_number




def get_CDCL_predicted_res(clauses):
    # 调用 Minisat22
    with Minisat22(bootstrap_with=clauses) as m:
        mini_sat = m.solve()


        stats = m.accum_stats()
        mini_branches_number = stats.get('decisions', 0)

    return mini_sat, mini_branches_number


# 主循环
for alpha in alpha_values:
    time_list_gpt, branches_list_gpt, sat_count_gpt = [], [], 0
    branches_list_mini = []
    mini_sat_count, correct_count = 0, 0
    L = int(alpha * N)
    unsat_cnt = 0
    file_num = 0

    # step 0: get the predicted res for each alpha
    for inst_idx in range(1, instances_per_alpha + 1):

        filename = f"k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}.cnf"
        filename_SAT = f"k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}_SAT.cnf"
        filename_UNSAT = f"k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}_UNSAT.cnf"

        filepath_SAT = os.path.join(input_dir, filename_SAT)
        filepath_UNSAT = os.path.join(input_dir, filename_UNSAT)

        if os.path.exists(filepath_SAT):
            filepath = filepath_SAT
        elif os.path.exists(filepath_UNSAT):
            filepath = filepath_UNSAT
        else:
            print("The file does not exist")


        if "k3_N75_L300_alpha4.0_inst270_SAT" in filepath:
            print(filepath)

        # step 1: get GPT prediction res
        time_val, branches_number, conflicts_number, gpt_sat, clauses = extract_info_from_file(filepath)
        gpt_sat, branches_number, conflicts_number = set_value_for_specific_files(filepath, gpt_sat, branches_number, conflicts_number)

        if isinstance(branches_number, list):
            print("filepath--:", filepath)
            print("gpt_sat:", gpt_sat)
            print("branches_number:", branches_number)

        # step 2: get CDCL predicted res
        mini_sat, mini_branches_number = get_CDCL_predicted_res(clauses)

        # step 3: compare GPT and CDCL res
        if gpt_sat is  None or branches_number is  None:
            continue

        if gpt_sat != mini_sat:
            correct_flag = False
        else:
            correct_flag = True
            correct_count += 1
            continue

        if gpt_sat:
            sat_count_gpt += 1
        if mini_sat:
            mini_sat_count += 1





        # step 4: collect predicted res
        time_list_gpt.append(time_val)
        branches_list_gpt.append(branches_number)
        branches_list_mini.append(mini_branches_number)






    mean_branches_gpt.append(mean(branches_list_gpt) if branches_list_gpt else 0)
    median_branches_gpt.append(median(branches_list_gpt) if branches_list_gpt else 0)
    mean_branches_mini.append(mean(branches_list_mini) if branches_list_mini else 0)
    median_branches_mini.append(median(branches_list_mini) if branches_list_mini else 0)

    avg_times_gpt.append(mean(time_list_gpt) if time_list_gpt else 0)
    prob_sat_gpt.append(sat_count_gpt / len(branches_list_gpt) if branches_list_gpt else 0)
    prob_sat_mini.append(mini_sat_count / len(branches_list_gpt) if branches_list_gpt else 0)


    gpt_vs_mini_accuracy.append(correct_count / len(branches_list_gpt) if branches_list_gpt else 0)







if gpt_correct_flag_selected:

    # figure 1
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, mean_branches_gpt,marker='o', label='Mean branches', color='red')
    ax1.plot(alpha_values, median_branches_gpt,  '--', marker='s', label='Median branches', color='yellow')
    ax1.set_xlabel('L / N')
    ax1.set_ylabel('Number of branches')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat_gpt, '--', marker='^', color='blue', label='Prob(sat)')
    ax2.set_ylabel('Prob(sat)')

    plt.title("GPT " + model_selected + ": Phase Transition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figure, "N_75_GPT_" + model_selected + 'Phase Transition.png'))
    plt.show()



    # figure 2
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, mean_branches_mini, marker='o', label='Mean branches', color='black')
    ax1.plot(alpha_values, median_branches_mini, '--', marker='s', label='Median branches', color='black')
    ax1.set_xlabel('L / N')
    ax1.set_ylabel('Number of branches')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat_mini, '--',marker='^',  color='black', label='Prob(sat)')
    ax2.set_ylabel('Prob(sat)')

    plt.title('Random 3-SAT, CDCL, N = 75')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figure, 'Random_3-SAT_CDCL_N_75'+'_Phase_Transition.png'))
    plt.show()


    # figure 3
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alpha_values, mean_branches_gpt, marker='o', label='Mean branches', color='red')
    ax1.plot(alpha_values, median_branches_gpt, '--', marker='s', label='Median branches', color='yellow')
    ax1.plot(alpha_values, mean_branches_mini, marker='o', label='Mean branches', color='black')
    ax1.plot(alpha_values, median_branches_mini,  '--', marker='s', label='Median branches', color='black')
    ax1.set_xlabel('L / N')
    ax1.set_ylabel('Number of branches')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, prob_sat_gpt, '--', marker='^', color='blue', label='Prob(sat)')
    ax2.plot(alpha_values, prob_sat_mini, '--', marker='^', color='black', label='Prob(sat)')
    ax2.set_ylabel('Prob(sat)')

    plt.title("GPT " + model_selected + ": Phase Transition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figure, "N_75_GPT_" + model_selected + '_and_CDCL_Phase Transition.png'))
    plt.show()


else:
    # ===================
    # 图1：GPT vs Minisat SAT 概率对比图
    # ===================
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, prob_sat_gpt, '-o', label="GPT" + model_selected +" SAT Prob", color='orange')
    plt.plot(alpha_values, prob_sat_mini, '-s', label="CDCL SAT Prob", color='blue')
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
    plt.plot(alpha_values, mean_branches_gpt, '-o', label="Mean branches (GPT)", color='red')
    plt.plot(alpha_values, median_branches_gpt, '--s', label="Median branches (GPT)", color='yellow')
    plt.xlabel("L / N")
    plt.ylabel("Branches (GPT estimated)")
    plt.title("GPT " + model_selected + ": Mean & Median Branches")
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
    plt.savefig(os.path.join(output_dir_figure, "plot_accuracy_gpt_vs_minisat.png"))
    plt.show()







