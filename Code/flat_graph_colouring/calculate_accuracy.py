import os
import re
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef
from collections import Counter

# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.6f}'.format)


# ---------- 新的工具函数：解析 flat30-60_openai_prediction_gpt-5 目录下的文件 ----------
def extract_info_from_file_flat_graph(filepath: str):
    """
    解析 flat30-60_openai_prediction_gpt-5 目录下的单个结果文件。

    返回：
        time_val: float 或 None, 从 'c GPT solve time: xxx seconds' 解析
        answer: str 或 None, JSON 中的 "answer" 字段（一般为 "SAT" 或 "UNSAT"）
        error_signal: bool, 若 JSON 解析失败等则为 True
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    time_val = None
    error_signal = False
    answer = None

    # 解析 GPT solve time
    m_time = re.search(r"GPT solve time:\s*([0-9.]+)\s*seconds", text)
    if m_time:
        try:
            time_val = float(m_time.group(1))
        except ValueError:
            time_val = None

    # 解析最后一行 JSON（包含 "answer" 字段）
    # 某些文件可能在最后有 shell 提示符，这里尽量从文本里找出一个 JSON 对象
    m_json = re.search(r'\{.*"answer".*?\}', text, re.DOTALL)
    if not m_json:
        # 如果没找到，标记 error
        error_signal = True
        return time_val, answer, error_signal

    json_str = m_json.group(0)
    try:
        data = json.loads(json_str)
        answer = data.get("answer", None)
    except Exception:
        error_signal = True

    return time_val, answer, error_signal


# ---------- 你的指标函数：基本保持不变 ----------

def get_metrics_SAT(instances_res, root_dir, model_select, perspective='SAT'):
    grouped_by_N = defaultdict(list)

    # 分组
    for item in instances_res:
        N = item["N"]
        label = item["label_IS_SAT"]
        pred = item["prediction_IS_SAT"]
        grouped_by_N[N].append((label, pred))

    metrics = []

    for N, data in grouped_by_N.items():
        labels = [x[0] for x in data]
        preds = [x[1] for x in data]

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        metrics.append({
            "N": N,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "TP(SAT-SAT)": tp,
            "TN(UNSAT-UNSAT)": tn,
            "FP(UNSAT-SAT)": fp,
            "FN(SAT-UNSAT)": fn
        })

    df = pd.DataFrame(metrics)
    df = df.sort_values(by="N")

    analysis_dir = os.path.join(os.path.dirname(root_dir), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, model_select + "_metrics_mul_N_perspective_" + str(perspective) + ".xlsx")
    df.to_excel(out_path, index=False)
    print("Saved:", out_path)


def get_metrics_UNSAT(instances_res, root_dir, model_select, perspective='UNSAT'):
    grouped_by_N = defaultdict(list)

    for item in instances_res:
        N = item["N"]
        label = item["label_IS_SAT"]
        pred = item["prediction_IS_SAT"]
        grouped_by_N[N].append((label, pred))

    metrics = []

    for N, data in grouped_by_N.items():
        labels = [x[0] for x in data]
        preds = [x[1] for x in data]

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        metrics.append({
            "N": N,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "TP(UNSAT-UNSAT)": tp,
            "TN(SAT-SAT)": tn,
            "FP(SAT-UNSAT)": fp,
            "FN(UNSAT-SAT)": fn
        })

    df = pd.DataFrame(metrics)
    df = df.sort_values(by="N")

    analysis_dir = os.path.join(os.path.dirname(root_dir), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, model_select + "_metrics_mul_N_perspective_" + str(perspective) + ".xlsx")
    df.to_excel(out_path, index=False)
    print("df UNSAT category:", df)
    print("Saved:", out_path)



from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import pandas as pd
import os
import json
import re


def get_one_model_one_version_result(Predictions_before, Predictions_after):
    """
    before  : 对应 UNSAT 公式（真实标签 0）
    after   : 对应 SAT 公式（真实标签 1）

    Predictions_before / Predictions_after 是“是否预测为 SAT”的布尔值（或 0/1）：
        True/1  -> 预测为 SAT
        False/0 -> 预测为 UNSAT
    """
    # 真实标签：UNSAT = 0, SAT = 1
    y_labels_before = [0] * len(Predictions_before)
    y_labels_after = [1] * len(Predictions_after)

    one_row = {}
    y_true = y_labels_before + y_labels_after
    y_pred = Predictions_before + Predictions_after

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
        pb = int(Predictions_before[i])
        pa = int(Predictions_after[i])

        yb = y_labels_before[i]  # 0
        ya = y_labels_after[i]   # 1

        # before (UNSAT)
        if pb == yb:
            before_correct += 1
        else:
            before_wrong += 1

        # after (SAT)
        if pa == ya:
            after_correct += 1
        else:
            after_wrong += 1

        # 比较两个预测是否一致
        if pb == pa:
            same_prediction += 1
            if pb == 1:
                same_prediction_positive += 1
            else:
                same_prediction_negative += 1
        else:
            different_prediction += 1
            # “区分得正确”：两个都各自预测对了
            if pb == yb and pa == ya:
                different_prediction_correct += 1
            else:
                different_prediction_incorrect += 1

    DR = different_prediction / len(Predictions_before)
    ADR = different_prediction_correct / len(Predictions_before)
    SDR = 0.5 * DR + 0.5 * ADR
    CR = same_prediction / len(Predictions_before)

    DR = round(DR, 2)
    ADR = round(ADR, 2)
    SDR = round(SDR, 2)
    CR = round(CR, 2)

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
    one_row['Precision'] = precision
    one_row['Recall'] = recall
    one_row['F1'] = f1
    one_row['MCC'] = mcc
    one_row['ACC'] = acc

    print("one_row:", one_row)
    return one_row


def get_our_new_metrics(instances_res, model_select, root_dir):
    """
    适配 flat30-*-{SAT,UNSAT_UNSAT}_gpt-5.txt 的新版：

    - instances_res 中每一行包含：
        {
          "N": int,
          "file_name": str,
          "label_IS_SAT": bool,        # True->SAT, False->UNSAT
          "prediction_IS_SAT": bool,   # True->预测 SAT, False->预测 UNSAT
          ...
        }

    - 同一个 (N, id) 有 2 个文件：
        flat30-91_SAT_gpt-5.txt
        flat30-91_UNSAT_UNSAT_gpt-5.txt

    - 对于每个 N：
        - before:  所有 UNSAT 文件的 prediction_IS_SAT（label=0）
        - after:   所有 SAT   文件的 prediction_IS_SAT（label=1）

      组成 pair 后，计算 DR / ADR / SDR / CR / MCC。
    """
    all_rows = []
    N_List = sorted(list(set([item["N"] for item in instances_res])))

    # 方便按 file_name 查找
    instances_res_sorted = sorted(instances_res, key=lambda x: x['file_name'])

    # 保存每个 N 下、每个 id 的 pair 预测结果
    pairs_prediction_res = {}

    # 解析 id 的正则：flat30-91_SAT_gpt-5.txt -> id = "91"
    id_pattern = re.compile(r'flat\d+-(\d+)_')

    for N in N_List:
        # key: id_str -> {'SAT': pred, 'UNSAT': pred}
        id_to_preds = {}

        for item in instances_res_sorted:
            if item["N"] != N:
                continue

            fname = item["file_name"]
            m = id_pattern.search(fname)
            if not m:
                # 不符合命名模式则跳过
                continue

            id_str = m.group(1)

            # 初始化
            if id_str not in id_to_preds:
                id_to_preds[id_str] = {}

            pred = bool(item["prediction_IS_SAT"])
            is_sat_label = bool(item["label_IS_SAT"])  # True=SAT, False=UNSAT

            # 用真实标签来决定放哪一边（更稳，比看文件名字符串）
            if is_sat_label:
                id_to_preds[id_str]['SAT'] = pred
            else:
                id_to_preds[id_str]['UNSAT'] = pred

        # 把成对的 id 抽出来
        Predictions_before = []  # UNSAT -> label 0
        Predictions_after = []   # SAT   -> label 1
        one_N_pairs_prediction_res = {}  # 记录每个 id 的 before/after 预测

        for id_str, pred_dict in id_to_preds.items():
            if 'SAT' in pred_dict and 'UNSAT' in pred_dict:
                p_before = pred_dict['UNSAT']
                p_after = pred_dict['SAT']

                Predictions_before.append(p_before)
                Predictions_after.append(p_after)

                one_N_pairs_prediction_res[id_str] = {
                    'Prediction_before_UNSAT': p_before,
                    'Prediction_after_SAT': p_after
                }
            else:
                # 只有一个文件，不是完整 pair，跳过
                continue

        if len(Predictions_before) == 0:
            print(f"No complete pairs found when N = {N}")
            continue

        # 计算本 N 下的 ADR, MCC 等
        one_row = get_one_model_one_version_result(Predictions_before, Predictions_after)
        one_row["N"] = N
        one_row["model_select"] = model_select
        all_rows.append(one_row)

        pairs_prediction_res[N] = one_N_pairs_prediction_res

    # 保存 ADR/MCC 表
    all_rows_df = pd.DataFrame(all_rows)
    analysis_dir = os.path.join(os.path.dirname(root_dir), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    metrics_path = os.path.join(analysis_dir, model_select + "_our_new_metrics_flat_pairs.xlsx")
    all_rows_df.to_excel(metrics_path, index=False)
    print("Saved ADR/MCC metrics to:", metrics_path)

    # 保存 pair-level 预测结果
    pairs_dir = os.path.join(analysis_dir, 'pairs_prediction_res')
    os.makedirs(pairs_dir, exist_ok=True)
    original_pairs_prediction_res_path = os.path.join(
        pairs_dir,
        model_select + "_original_pairs_prediction_res_flat_pairs.json"
    )
    with open(original_pairs_prediction_res_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_prediction_res, f, ensure_ascii=False, indent=4)
        print("original_pairs_prediction_res_path:", original_pairs_prediction_res_path)



# 如果后面你还要用到 SDR / ADR，可以保留你原来的 get_one_model_one_version_result / get_our_new_metrics
# 这里先不动那一部分，因为 flat30-60 这套文件没有 before/after pair。


import os
import re
import pandas as pd


def parse_directory_results(root_dir: str):
    """
    遍历 root_dir 下所有 .txt 文件，解析：
      - 真实标签 label_IS_SAT（文件名含 UNSAT -> False，否则 True）
      - 预测标签 prediction_IS_SAT（来自 JSON answer）
      - N（从文件名 flat30-97_... 中解析 30）
      - time（解析 GPT solve time）

    返回:
        instances_res: List[dict]
        error_files:   List[str] 解析失败的文件路径
    """
    # from your_module import extract_info_from_file_flat_graph  # 如已在同文件，可直接删这一行

    instances_res = []
    error_files = []

    for file_name in os.listdir(root_dir):
        if not file_name.endswith('.txt'):
            continue

        file_path = os.path.join(root_dir, file_name)

        # 1) 真实标签：看文件名有没有 "UNSAT"
        if 'UNSAT' in file_name:
            label_IS_SAT = False  # 真实是 UNSAT
        else:
            label_IS_SAT = True   # 真实是 SAT

        # 2) 解析文件内部 JSON 的 "answer"
        time_val, answer, error_signal = extract_info_from_file_flat_graph(file_path)

        if error_signal or answer is None:
            print("解析失败, 请检查文件:", file_path)
            error_files.append(file_path)
            continue

        ans_upper = str(answer).strip().upper()
        if ans_upper == 'SAT':
            prediction_IS_SAT = True
        elif ans_upper == 'UNSAT':
            prediction_IS_SAT = False
        else:
            print("未知 answer 值:", answer, "in file:", file_path)
            error_files.append(file_path)
            continue

        # 3) 提取 N（例如 flat30-97_SAT_gpt-5.txt -> 30）
        m_n = re.search(r'flat(\d+)-', file_name)
        if m_n:
            N_val = int(m_n.group(1))
        else:
            # 若解析不到，就给一个默认值（根据你的数据可调整）
            N_val = 30

        instances_res.append({
            "dir": root_dir,
            "N": N_val,
            "file_name": file_name,
            "label_IS_SAT": label_IS_SAT,
            "prediction_IS_SAT": prediction_IS_SAT,
            "time": time_val
        })

    return instances_res, error_files


def save_instances_results(instances_res, root_dir: str, model_select: str):
    """
    将 instances_res 排序后保存为 Excel，便于后续查看。
    """
    analysis_dir = os.path.join(os.path.dirname(root_dir), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    instances_res_sorted = sorted(instances_res, key=lambda x: (x["N"], x["file_name"]))
    df = pd.DataFrame(instances_res_sorted)

    out_path = os.path.join(analysis_dir, model_select + "_instances_res.xlsx")
    df.to_excel(out_path, index=False)
    print("Saved instances_res to:", out_path)


def get_metrics_SAT(instances_res, root_dir, model_select, perspective = 'SAT' ):
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
            "N": N,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "TP(SAT-SAT)": tp,
            "TN(UNSAT-UNSAT)": tn,
            "FP(UNSAT-SAT)": fp,
            "FN(SAT-UNSAT)": fn
        })

    # 转为 DataFrame
    df = pd.DataFrame(metrics)
    df = df.sort_values(by="N")  # 可选排序

    print("df:", df)
    # 保存为 Excel
    df.to_excel(os.path.join(os.path.dirname(root_dir), 'analysis', model_select + "_metrics_mul_N_perspective_"+str(perspective)+".xlsx"), index=False)
    print("Saved to metrics_with_confusion_matrix.xlsx")



def get_metrics_UNSAT(instances_res, root_dir, model_select, perspective = 'SAT' ):
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
            "N": N,
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
    df.to_excel(os.path.join(os.path.dirname(root_dir), 'analysis', model_select + "_metrics_mul_N_perspective_"+str(perspective)+".xlsx"), index=False)
    print("Saved to metrics_with_confusion_matrix.xlsx")


def compute_and_save_metrics_for_both_classes(instances_res, root_dir: str, model_select: str):
    """
    先以 SAT 为正类计算一次指标，
    再将标签和预测全部取反，以 UNSAT 为正类再算一次。
    """
    # from your_module import get_metrics_SAT, get_metrics_UNSAT  # 若在同文件，可删掉这一行

    # 以 SAT 为正类
    get_metrics_SAT(instances_res, root_dir, model_select, perspective='SAT')

    # 为计算 UNSAT 视角，把标签和预测全部翻转
    for item in instances_res:
        item["label_IS_SAT"] = not item["label_IS_SAT"]
        item["prediction_IS_SAT"] = not item["prediction_IS_SAT"]

    get_metrics_UNSAT(instances_res, root_dir, model_select, perspective='UNSAT')


def analyze_flat_graph_directory(root_dir: str, model_select: str = 'gpt-5'):
    """
    顶层封装函数：给定一个目录（flat30-60_openai_prediction_gpt-5），
    一次性完成：
      1) 解析所有文件
      2) 保存 instances_res 结果表
      3) 计算 SAT / UNSAT 两个视角的指标
      4) 打印解析失败的文件列表
    """
    # 1. 解析目录
    instances_res, error_files = parse_directory_results(root_dir)

    # 2. 保存详细结果
    save_instances_results(instances_res, root_dir, model_select)

    # 3. 计算并保存指标
    compute_and_save_metrics_for_both_classes(instances_res, root_dir, model_select)

    get_our_new_metrics(instances_res, model_select, root_dir)
    # 4. 打印解析失败文件
    if error_files:
        print("\n解析失败的文件数量:", len(error_files))
        for ef in error_files:
            print("  ", ef)


if __name__ == '__main__':
    root_dir = '/scratch/lzhan011/Satisfiability_Solvers/Code/flat_graph_colouring/Flat_Graph_Colouring/flat30-60_openai_prediction_gpt-5'
    model_select = 'gpt-5'

    analyze_flat_graph_directory(root_dir, model_select)




