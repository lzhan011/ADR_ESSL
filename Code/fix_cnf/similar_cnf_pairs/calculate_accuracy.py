import os
import re
import pandas as pd
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef
from collections import Counter
import json

# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行（如需要）
pd.set_option('display.max_rows', None)

# 设置列内容不省略
pd.set_option('display.max_colwidth', None)

# 禁用科学计数法（可选）
pd.set_option('display.float_format', '{:.6f}'.format)



# ---------- 工具函数 ----------
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
        if line.startswith("p cnf"):
            read_clause = True
            continue
        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)

        line = line.replace("boxed", "").replace("text", "").replace("Result", "")
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
        elif 'SATISFIABLE: UNSATISFIABLE' in line.upper():
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
        print(filepath)



        sat_index_next = lines[sat_index_next].strip()
        if len(sat_index_next) == 0:
            sat_index = sat_index + 1

        try:
            branches_index = sat_index + 1
            branches_number = lines[branches_index]
            branches_number = re.findall(r'\d*', branches_number)
            branches_number = int(''.join(branches_number))

            conflicts_index = sat_index + 2
            conflicts_number = lines[conflicts_index]
            conflicts_number = re.findall(r'\d*', conflicts_number)
            conflicts_number = int(''.join(conflicts_number))
        except:
            print("please check the file:", filepath)


    return time_val, branches_number, conflicts_number, sat, clauses, error_signal





def get_metrics_SAT(instances_res, root_dir, model_select, perspective = 'SAT', one_level_keyword="" ):
    grouped_by_N = defaultdict(list)

    # 分组
    for item in instances_res:
        N = item["N"]
        label = item["label_IS_SAT"]
        pred = item["prediction_IS_SAT"]
        grouped_by_N[N].append((label, pred))

    # 存储所有结果
    metrics = []

    for N, data in grouped_by_N.items():
        labels = [x[0] for x in data]
        preds = [x[1] for x in data]

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        # 指标
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # 添加到结果列表
        metrics.append({
            "model": model_select,
            "N": N,
            "similar_level": one_level_keyword,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "TP(SAT-SAT)": tp,
            "TN(UNSAT-UNSAT)": tn,
            "FP(UNSAT-SAT)": fp,
            "FN(SAT-UNSAT)": fn,
        })

    # 转为 DataFrame
    df = pd.DataFrame(metrics)
    df = df.sort_values(by="N")  # 可选排序

    # 保存为 Excel
    df.to_excel(os.path.join(root_dir, 'analysis', model_select + "_metrics_mul_N_perspective_"+str(perspective)+"_"+one_level_keyword+".xlsx"), index=False)
    print("Saved to metrics_with_confusion_matrix.xlsx")




def get_metrics_UNSAT(instances_res, root_dir, model_select, perspective = 'SAT', one_level_keyword="" ):
    grouped_by_N = defaultdict(list)

    # 分组
    for item in instances_res:
        N = item["N"]
        label = item["label_IS_SAT"]
        pred = item["prediction_IS_SAT"]
        grouped_by_N[N].append((label, pred))

    # 存储所有结果
    metrics = []

    for N, data in grouped_by_N.items():
        labels = [x[0] for x in data]
        preds = [x[1] for x in data]

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        # 指标
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # 添加到结果列表
        metrics.append({
            "model":model_select,
            "N": N,
            "similar_level":one_level_keyword,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "TP(UNSAT-UNSAT)": tp,
            "TN(SAT-SAT)": tn,
            "FP(SAT-UNSAT)": fp,
            "FN(UNSAT-SAT)": fn
        })

    # 转为 DataFrame
    df = pd.DataFrame(metrics)
    df = df.sort_values(by="N")  # 可选排序

    # 保存为 Excel
    df.to_excel(os.path.join(root_dir, 'analysis', model_select + "_metrics_mul_N_perspective_"+str(perspective)+"_"+one_level_keyword+".xlsx"), index=False)
    print("Saved to metrics_with_confusion_matrix.xlsx")



def get_one_model_one_version_result(Predictions_before, Predictions_after):
    y_labels_before = [0] * len(Predictions_before)
    y_labels_after =  [1] * len(Predictions_after)  # 1 represent is SAT, after fix, the cnf will be SAT
    one_row = {}
    y_true = y_labels_before + y_labels_after
    y_pred = Predictions_before + Predictions_after
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    before_correct = 0
    after_correct = 0
    before_wrong = 0
    after_wrong = 0

    same_prediction = 0
    different_prediction = 0
    same_prediction_positive = 0
    same_prediction_negative = 0

    different_prediction_correct = 0
    different_prediction_incorrect = 0
    for i in range(len(Predictions_before)):
        Predictions_before_i = Predictions_before[i]
        y_labels_before_i = y_labels_before[i]

        Predictions_after_i = Predictions_after[i]
        y_labels_after_i = y_labels_after[i]
        if Predictions_before_i == y_labels_before_i:
            before_correct += 1
        else:
            before_wrong += 1

        if Predictions_after_i == y_labels_after_i:
            after_correct += 1
        else:
            after_wrong += 1

        if Predictions_before_i == Predictions_after_i:
            same_prediction += 1

            if Predictions_before_i:
                same_prediction_positive += 1
            else:
                same_prediction_negative += 1

        else:
            different_prediction += 1

            if Predictions_before_i == y_labels_before_i and Predictions_after_i == y_labels_after_i:
                different_prediction_correct += 1
            else:
                different_prediction_incorrect += 1

    # print("before_correct:", before_correct)
    # print("before_wrong:", before_wrong)
    # print("after_correct:", after_correct)
    # print("after_wrong:", after_wrong)
    #
    # print("same_prediction:", same_prediction)
    # print("different_prediction:", different_prediction)
    # print("same_prediction_positive:", same_prediction_positive)
    # print("same_prediction_negative:", same_prediction_negative)
    # print("different_prediction_correct:", different_prediction_correct)
    # print("different_prediction_incorrect:", different_prediction_incorrect)

    DR = different_prediction / len(Predictions_before)
    ADR = different_prediction_correct / len(Predictions_before)
    SDR = 0.5 * DR + 0.5 * ADR
    CR = same_prediction / len(Predictions_before)

    DR = round(DR, 2)
    ADR = round(ADR, 2)
    SDR = round(SDR, 2)
    CR = round(CR, 2)

    # print("DR:", DR)
    # print("ADR:", ADR)
    # print("SDR:", SDR)
    # print("CR:", CR)
    one_row['C (Number of Confused)'] = same_prediction
    one_row['CP (Number of Confused-positive)'] = same_prediction_positive
    one_row['CN (Number of Confused-negative)'] = same_prediction_negative
    one_row['S(Number of Separated)'] = different_prediction
    one_row['SC (Number of Separated-correct)'] = different_prediction_correct
    one_row['SI (Number of Separated-incorrect)'] = different_prediction_incorrect
    one_row['DR (Differentiation Rate)'] = DR
    one_row['ADR (Accurate Differentiation Rate)'] = ADR
    one_row['SDR (Symmetric Differentiation Rate)'] = SDR
    one_row['CR (Confusion Rate)'] = CR
    one_row['MCC'] = mcc
    print("one_row:", one_row)
    return one_row



def get_our_new_metrics(instances_res, model_select, root_dir, one_level_keyword=""):
    all_rows = []
    N_List = sorted(list(set([item["N"] for item in instances_res])))
    tail_trim = -9
    file_name_list = sorted(list([item["file_name"].replace("_RC2_fixed", "")[:tail_trim] for item in instances_res]))
    # 统计每个 file_name 出现的次数
    counter = Counter(file_name_list)
    # 只保留那些出现两次的 file_name
    file_name_list_filtered = [name for name in file_name_list if counter[name] == 2]

    instances_res.sort(key=lambda x: x['file_name'], reverse=True)

    if model_select == 'deepseek-reasoner':
        N_List = [5, 8, 10, 25, 50]

    pairs_prediction_res = {}
    for N in N_List:
        one_N_pairs_prediction_res = {item:{} for item in file_name_list_filtered}
        Predictions_after_fix = []
        Predictions_before_fix = []
        for item in instances_res:
            if N == item["N"]:
                file_name = item["file_name"]
                file_name_raw = file_name.replace("_RC2_fixed", "")[:tail_trim]
                if file_name_raw  not in file_name_list_filtered:
                    continue

                prediction_IS_SAT = item["prediction_IS_SAT"]
                if "fixed" in file_name:
                    Predictions_after_fix.append(prediction_IS_SAT)
                    one_N_pairs_prediction_res[file_name_raw]['Predictions_after_fix'] = prediction_IS_SAT
                else:
                    Predictions_before_fix.append(prediction_IS_SAT)
                    one_N_pairs_prediction_res[file_name_raw]['Predictions_before_fix'] = prediction_IS_SAT

        one_row = get_one_model_one_version_result(Predictions_before_fix, Predictions_after_fix)
        one_row['model'] = model_select
        one_row["N"] = N
        one_row["model_select"] = model_select
        one_row['similar_level'] = one_level_keyword
        all_rows.append(one_row)
        pairs_prediction_res[N] = one_N_pairs_prediction_res
    all_rows = pd.DataFrame(all_rows)
    all_rows.to_excel(os.path.join(root_dir, 'analysis', model_select +"_"+one_level_keyword+ "_our_new_metrics.xlsx"), index=False)

    with open(os.path.join(root_dir, 'analysis/pairs_prediction_res', model_select +"_"+one_level_keyword+ "_pairs_prediction_res.json"), 'w', encoding='utf-8' ) as f:
        json.dump(pairs_prediction_res, f, ensure_ascii=False, indent=4)

    return pairs_prediction_res


def search_level_keywords(filename, level_keywords):

    found = [kw for kw in level_keywords if kw in filename]
    if len(found) == 1:
        return found[0]
    else:
        return ''


# if __name__ == '__main__':
#
#
#     y_labels_before  = [1,1,1,1]
#     y_labels_after  = [0,0,0,0]
#     Predictions_before = [1,1,0,0]
#     Predictions_after = [0,0,1,1]
#
#     results = get_one_model_one_version_result(Predictions_before, Predictions_after, y_labels_before, y_labels_after)
#
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
#
#
# exit()

if __name__ == '__main__':
    # gpt-3.5-turbo-0125
    instances_res  = []
    model_select =  "gpt-5" #"gpt-4.1" #"gpt-4-turbo" #"gpt-3.5-turbo-0125" #"gpt-3.5-turbo" #"gpt-4o-latest"  #  gpt-3.5-turbo   "o1" "openai_prediction"   "o3-mini"
    # root_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_traditional_methond'

    model_select =  'gpt-4-turbo' #"gpt-5"# "deepseek-reasoner"#"gpt-3.5-turbo" #"gpt-4.1" #"gpt-4o"#"gpt-4-turbo"#"o3-mini"#"deepseek-reasoner" #"deepseek-reasoner" #'deepseek-reasoner' #'o1'#'o3-mini' #"o1"#'deepseek-reasoner' #'chatgpt-4o-latest' #'gpt-3.5-turbo'#'gpt-3.5-turbo-0125' #'gpt-4o' #"gpt-4-turbo" #'o3-mini'
    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result'
    level_keywords = ["low", "medium", "high", ]
    # root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/'
    error_count = []
    unmatched_res_pattern = []
    for dir in os.listdir(root_dir):

        dir_path = os.path.join(root_dir, dir)

        if model_select == 'o1':
            if not dir_path.endswith('o1_openai_prediction_o1'):
                continue

        # if dir_path.endswith("openai_prediction"):
        if dir_path.endswith(model_select):

            print("\n\ndir:", dir)
            unsat_cnt = 0
            file_num = 0

            for file_name in os.listdir(dir_path):

                level_keyword = search_level_keywords(file_name, level_keywords)


                if 'cnf_k3_N10_L35_alpha3.5_inst793_RC2_fixed_low_1_' in file_name:
                    print(file_name)

                if 'RC2_fixed' in file_name:
                    label_IS_SAT = True
                else:
                    label_IS_SAT = False

                file_path = os.path.join(dir_path, file_name)
                time_val, branches_number, conflicts_number, prediction_IS_SAT, clauses, error_signal = extract_info_from_file(file_path, model_select)

                if error_signal:
                    print("file_path:", file_path)
                    error_count.append(error_signal)
                    print("error_count:", len(error_count))
                    continue

                if prediction_IS_SAT is None:
                    print("file_path:",file_path)
                    print("prediction_IS_SAT:", prediction_IS_SAT)
                    unmatched_res_pattern.append(file_path)
                    print("unmatched_res_pattern:", len(unmatched_res_pattern))
                    continue

                dir_n = re.findall(r"N(\d+)", file_name)[0]
                instances_res.append({    "dir": dir,
                                          "N": int(dir_n),
                                          "file_name": file_name,
                                          "level_keyword":level_keyword,
                                          "label_IS_SAT": label_IS_SAT,
                                          "prediction_IS_SAT": prediction_IS_SAT,
                                          })
            instances_res.sort(key=lambda x: x["N"])
    instances_res_df = pd.DataFrame(instances_res)
    instances_res_df.to_excel(os.path.join(root_dir, 'analysis', model_select + "_instances_res_cross_n.xlsx"))
    # instances_res.to_excel(os.path.join(root_dir, "correct_rate_cross_n_GPT-4.1_V2.xlsx"))

    for one_level_keyword in level_keywords:

        print("one_level_keyword:", one_level_keyword)
        one_level_keyword_instances_res = [item for item in instances_res if item["level_keyword"] == one_level_keyword]

        get_metrics_SAT(one_level_keyword_instances_res, root_dir, model_select, perspective='SAT', one_level_keyword=one_level_keyword)

        pairs_prediction_res = get_our_new_metrics(one_level_keyword_instances_res, model_select, root_dir, one_level_keyword=one_level_keyword)


        # 将UNSAT 设为 计算的对象
        for item in one_level_keyword_instances_res:
            item["label_IS_SAT"] = not item["label_IS_SAT"]
            item["prediction_IS_SAT"] = not item["prediction_IS_SAT"]
        get_metrics_UNSAT(one_level_keyword_instances_res, root_dir, model_select, perspective='UNSAT', one_level_keyword=one_level_keyword)






