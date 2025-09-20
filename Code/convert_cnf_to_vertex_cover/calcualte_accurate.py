from convert_cnf_to_vertex_cover_method_1 import  *
import pandas as pd
import time
import os
import networkx as nx  # 用于构造重标号后的图 H（与 prompt 一致）
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef

# 重标号成整数图 H（1..|V|），用于校验 LLM 返回的 cover
def make_relabelled_graph(G: nx.Graph, mapping: dict) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(mapping.values())
    for u, v in G.edges():
        H.add_edge(mapping[u], mapping[v])
    return H

def evaluate(INPUT_CNF_ROOT, OUTPUT_ROOT, model_selected: str, N: int):
    dir_name = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
    input_dir  = os.path.join(INPUT_CNF_ROOT, dir_name)
    output_dir = os.path.join(OUTPUT_ROOT, dir_name) + '_openai_prediction_' + str(model_selected)
    prompt_dir = os.path.join(output_dir, "vc_prompts")
    answers_dir = os.path.join(prompt_dir, "answers")

    if not os.path.isdir(answers_dir):
        return []

    rows = []
    for fname in sorted(os.listdir(answers_dir)):
        if not fname.endswith(f"_llm_answer_{model_selected.replace('/','_')}.txt"):
            continue

        inst_base = fname[: -len(f"_llm_answer_{model_selected.replace('/','_')}.txt")]
        if 'RC2_fixed' in inst_base:
            cnf_path  = os.path.join(input_dir, inst_base + ".cnf")
        else:
            cnf_path = os.path.join(input_dir, inst_base + ".txt")
        ans_path  = os.path.join(answers_dir, fname)

        # if not os.path.isfile(cnf_path):
        #     continue

        # 1) CNF -> 3CNF -> VC (G,k)  （仍用于 validated 指标的真值）
        clauses_int   = read_dimacs(cnf_path)
        clauses_str_3 = int_clauses_to_str_3cnf(clauses_int)
        G, k, _       = reduce_3cnf_to_vertex_cover(clauses_str_3)
        gt_exists, _  = vc_decide_fast(G, k, use_sat=True)  # 算法真值（是否存在 VC≤k）

        # 2) 与 prompt 一致的重标号（整数 1..|V|）
        V, E, mapping = graph_to_sets_sorted(G, relabel_to_int=True)
        H              = make_relabelled_graph(G, mapping)

        # 3) 解析 LLM 输出
        with open(ans_path, "r", encoding="utf-8") as f:
            ans_text = f.read()
        ok_parse, llm_yes, cover_llm = parse_llm_answer_json(ans_text)

        # 4) 由**文件名**得到 answer 的真值（你的要求）
        #    文件名包含 "RC2_fixed" => YES（True），否则 NO（False）
        gt_from_name = ("RC2_fixed" in inst_base)

        # 5) 验证 cover（仅 YES 时）
        cover_valid = False
        cov_size    = None
        if ok_parse and llm_yes:
            cov_size    = len(cover_llm)
            cover_valid = (cov_size <= k) and is_vertex_cover(H, set(cover_llm))

        # 6) 两种“label 是否正确”的标记：
        #  - label_correct_algo：与算法真值比（保持原先逻辑，供参考/兼容）
        #  - label_correct_name：与文件名真值比（本次你要的指标依据）
        label_correct_algo = (ok_parse and (llm_yes == gt_exists))
        label_correct_name = (ok_parse and (llm_yes == gt_from_name))

        rows.append({
            "model": model_selected,
            "N": N,
            "instance": inst_base + ".cnf",
            "k": k,
            "|V|": len(V),
            "|E|": len(E),

            # 真值：算法版 & 文件名版
            "gt_exists": bool(gt_exists),         # 算法真值（用于 validated 指标）
            "gt_from_name": bool(gt_from_name),   # 文件名真值（用于 answer 指标）

            # LLM 解析/输出
            "llm_parsed": bool(ok_parse),
            "llm_answer_yes": (bool(llm_yes) if ok_parse else None),
            "llm_cover_size": cov_size,
            "cover_valid": bool(cover_valid),

            # 正确性（两套）
            "label_correct_algo": bool(label_correct_algo),
            "label_correct_name": bool(label_correct_name),

            # validated 正确：保持与算法真值对齐
            "validated_correct": (bool(label_correct_algo) and (cover_valid if (ok_parse and llm_yes) else True))
        })
    return rows

# ============== 指标计算：Precision/Recall/F1/ACC/Confusion ==============
def _confusion_counts(y_true, y_pred):
    import numpy as np
    yt = np.array(y_true, dtype=bool)
    yp = np.array(y_pred, dtype=bool)
    TP = int(( yt &  yp).sum())
    FP = int((~yt &  yp).sum())
    TN = int((~yt & ~yp).sum())
    FN = int(( yt & ~yp).sum())
    return TP, FP, TN, FN

def _metrics_from_confusion(TP, FP, TN, FN):
    total = TP + FP + TN + FN
    acc  = (TP + TN) / total if total else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, total

def compute_group_metrics(df_group, pred_col, true_col):
    """
    仅对解析成功的样本（llm_parsed==True）计算：
      - 混淆矩阵（TP,FP,TN,FN）
      - ACC / Precision / Recall / F1
    """
    sub = df_group[df_group["llm_parsed"] == True].copy()
    if len(sub) == 0:
        return {
            "parsed_total": 0,
            "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "ACC": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0
        }
    y_true = sub[true_col].astype(bool).tolist()
    y_pred = sub[pred_col].astype(bool).tolist()
    TP, FP, TN, FN = _confusion_counts(y_true, y_pred)
    ACC, PREC, REC, F1, total = _metrics_from_confusion(TP, FP, TN, FN)
    return {
        "parsed_total": total,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "ACC": ACC, "Precision": PREC, "Recall": REC, "F1": F1
    }

def main():
    # ---------------------- 配置 ----------------------
    INPUT_CNF_ROOT = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    OUTPUT_ROOT    = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/vertex_cover_graph'
    ANALYSIS_DIR   = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis'
    MODEL_LIST = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', 'gpt-4o', 'o3-mini', 'deepseek-reasoner','o1', 'gpt-5']
    N_LIST = [5, 8, 10, 25, 50, 60]
    # N_LIST = [50]
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    from tqdm.auto import tqdm

    all_rows = []
    for model in tqdm(MODEL_LIST, desc="Models"):
        for N in tqdm(N_LIST, desc=f"N for {model}", leave=False):
            rows = evaluate(INPUT_CNF_ROOT, OUTPUT_ROOT, model, N)
            all_rows.extend(rows)

    if not all_rows:
        print("没有找到任何 answers 文件可用于评测。请先生成 vc_prompts/answers。")
        return

    df = pd.DataFrame(all_rows)



    # ===== 原有 summary（保持不变：算法真值的整体表现）=====
    agg = df.groupby(["model","N"]).agg(
        total=("label_correct_algo","size"),
        label_acc=("label_correct_algo", "mean"),
        validated_acc=("validated_correct", "mean"),
        gt_positive_rate=("gt_exists", "mean"),
        parsed_rate=("llm_parsed","mean")
    ).reset_index()

    # ===== 新增：answer 指标（按你的要求：真值来自文件名）=====
    # y_true = gt_from_name, y_pred = llm_answer_yes
    metrics_answer_rows = []
    for (m, n), g in df.groupby(["model", "N"]):
        met = compute_group_metrics(g, pred_col="llm_answer_yes", true_col="gt_from_name")
        met.update({"model": m, "N": n})
        metrics_answer_rows.append(met)
    metrics_answer = pd.DataFrame(metrics_answer_rows)[
        ["model","N","parsed_total","TP","FP","TN","FN","ACC","Precision","Recall","F1"]
    ].sort_values(["model","N"])

    # ===== validated cover 指标（保持与算法真值对齐）=====
    # y_true = gt_exists, y_pred = cover_valid
    metrics_valid_rows = []
    for (m, n), g in df.groupby(["model", "N"]):
        met = compute_group_metrics(g, pred_col="cover_valid", true_col="gt_exists")
        met.update({"model": m, "N": n})
        metrics_valid_rows.append(met)
    metrics_validated = pd.DataFrame(metrics_valid_rows)[
        ["model","N","parsed_total","TP","FP","TN","FN","ACC","Precision","Recall","F1"]
    ].sort_values(["model","N"])

    # ===== 保存 Excel =====
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_xlsx = os.path.join(ANALYSIS_DIR, f"llm_vc_accuracy_{ts}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="details", index=False)
        agg.to_excel(writer, sheet_name="summary_algo_truth", index=False)
        metrics_answer.to_excel(writer, sheet_name="metrics_answer_by_name", index=False)
        metrics_validated.to_excel(writer, sheet_name="metrics_validated_algo_truth", index=False)
        # metrics_adr.to_excel(writer, sheet_name="ADR_by_pairs", index=False)

    print("保存结果：", out_xlsx)
    print("\n=== Summary (algo truth) ===")
    print(agg.to_string(index=False))
    print("\n=== Metrics (answer by filename truth) ===")
    print(metrics_answer.to_string(index=False))
    print("\n=== Metrics (validated cover vs algo truth) ===")
    print(metrics_validated.to_string(index=False))


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
    accuracy = accuracy_score(y_true, y_pred)
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
    one_row['precision'] = precision
    one_row['recall'] = recall
    one_row['f1'] = f1
    one_row['accuracy'] = accuracy
    print("one_row:", one_row)
    return one_row


def get_pairs(all_files):
    # --- 先按“成对”分组：key = 去掉后缀 _RC2_fixed 的文件基名 ---


    pairs = {}  # key -> {"orig": filename(无 _RC2_fixed), "fixed": filename(有 _RC2_fixed)}
    for f in all_files:
        stem = os.path.splitext(f)[0]
        if stem.endswith("_RC2_fixed"):
            key = stem[: -len("_RC2_fixed")]
            pairs.setdefault(key, {}).update({"fixed": f})
        else:
            key = stem
            pairs.setdefault(key, {}).update({"orig": f})

    return pairs


def compute_adr_by_model_N(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个 (model, N) 的 ADR（成对区分率）：
      - 以文件名去掉结尾的 "_RC2_fixed" 作为 pair-key
      - 每个 key 若同时存在原始与 _RC2_fixed 两个成员，计为一对
      - 用两种口径给出 ADR：
          1) ADR_by_label_correct：该对的两个成员的 label_correct_name 都为 True 记为成功
             （label_correct_name 已经把 parse 成功与否、是否与“文件名真值”一致都纳入了）
          2) ADR_by_llm_yes：orig 预测 NO 且 fixed 预测 YES 记为成功
             （分别由 llm_answer_yes=False / True 判定；若任一为 None 则视为失败）

    返回列：
      ['model','N','pairs_total','pairs_both_parsed',
       'pairs_success_label','pairs_success_llm_yes',
       'ADR_by_label_correct','ADR_by_llm_yes']
    """
    import os
    records = []

    # 辅助：归一化 pair-key 与成员类型
    def _pair_key(inst_name: str) -> str:
        stem = os.path.splitext(str(inst_name))[0]
        return stem[:-len("_RC2_fixed")] if stem.endswith("_RC2_fixed") else stem

    def _member_role(inst_name: str) -> str:
        stem = os.path.splitext(str(inst_name))[0]
        return "fixed" if stem.endswith("_RC2_fixed") else "orig"

    for (m, n), g in df.groupby(["model", "N"]):
        # 构造 pair 映射： key -> {'orig': row, 'fixed': row}
        pairs = {}
        for _, row in g.iterrows():
            key = _pair_key(row["instance"])
            role = _member_role(row["instance"])
            pairs.setdefault(key, {})[role] = row

        pairs_total = 0
        pairs_both_parsed = 0
        pairs_success_label = 0
        pairs_success_llm_yes = 0

        for key, d in pairs.items():
            if "orig" not in d or "fixed" not in d:
                continue  # 跳过不完整的一对
            ro, rf = d["orig"], d["fixed"]
            pairs_total += 1

            # 是否两端均解析成功（仅作统计参考，不决定分母）
            both_parsed = (ro.get("llm_parsed") is True) and (rf.get("llm_parsed") is True)
            if both_parsed:
                pairs_both_parsed += 1

            # 口径1：两端 label_correct_name 都为 True 记为成功
            if (ro.get("label_correct_name") is True) and (rf.get("label_correct_name") is True):
                pairs_success_label += 1

            # 口径2：orig 预测 NO，fixed 预测 YES 记为成功（None 视为失败）
            po = ro.get("llm_answer_yes", None)
            pf = rf.get("llm_answer_yes", None)
            if (po is not None) and (pf is not None) and (po is False) and (pf is True):
                pairs_success_llm_yes += 1

        adr_label = (pairs_success_label / pairs_total) if pairs_total else 0.0
        adr_llm   = (pairs_success_llm_yes / pairs_total) if pairs_total else 0.0

        records.append({
            "model": m,
            "N": n,
            "pairs_total": pairs_total,
            "pairs_both_parsed": pairs_both_parsed,
            "pairs_success_label": pairs_success_label,
            "pairs_success_llm_yes": pairs_success_llm_yes,
            "ADR_by_label_correct": adr_label,
            "ADR_by_llm_yes": adr_llm,
        })

    return pd.DataFrame(records).sort_values(["model", "N"])



def sort_df_res(all_N_all_model_rows):
    Nnum = (all_N_all_model_rows['N']
            .astype(str)
            .str.extract(r'(\d+)', expand=False)
            .astype('Int64'))
    all_N_all_model_rows = (all_N_all_model_rows
                            .assign(Nnum=Nnum)
                            .sort_values(by=['model', 'Nnum'])
                            .drop(columns='Nnum')
                            .reset_index(drop=True))

    return all_N_all_model_rows



def get_all_metrics():
    ANALYSIS_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis'

    all_N_all_model_rows = []
    for file in os.listdir(ANALYSIS_DIR):
        if "20250818" in file:
            N = int(file.split('_N_')[1].split('.')[0])  # 25

            one_N_df = pd.read_excel(os.path.join(ANALYSIS_DIR, file), sheet_name='details')
            one_N_df = one_N_df.dropna(subset=['llm_answer_yes'])
            model_list = one_N_df['model'].unique().tolist()
            for model in model_list:
                Predictions_before = []
                Predictions_after = []
                one_N_one_model_instance = one_N_df[one_N_df['model'] == model]
                pairs = get_pairs(one_N_one_model_instance['instance'].tolist())
                print(pairs)
                for pairs_key, pairs_v in pairs.items():

                    if "fixed" in pairs_v and "orig" in pairs_v:
                        instance_name_fixed = pairs[pairs_key]["fixed"]
                        instance_name_orig = pairs[pairs_key]["orig"]
                        instance_fixed_prediction = \
                        one_N_one_model_instance[one_N_one_model_instance['instance'] == instance_name_fixed][
                            'llm_answer_yes'].to_numpy().item()
                        instance_orig_prediction = \
                        one_N_one_model_instance[one_N_one_model_instance['instance'] == instance_name_orig][
                            'llm_answer_yes'].to_numpy().item()
                        Predictions_after.append(instance_fixed_prediction)
                        Predictions_before.append(instance_orig_prediction)
                one_row = get_one_model_one_version_result(Predictions_before, Predictions_after)
                one_row['model'] = model
                one_row['N'] = N

                all_N_all_model_rows.append(one_row)
    all_N_all_model_rows = pd.DataFrame(all_N_all_model_rows)
    all_N_all_model_rows.insert(0, 'model', all_N_all_model_rows.pop('model'))
    all_N_all_model_rows.insert(1, 'N', all_N_all_model_rows.pop('N'))
    # 把 N 提取为数值再排
    all_N_all_model_rows = sort_df_res(all_N_all_model_rows)
    all_N_all_model_rows.to_excel(os.path.join(ANALYSIS_DIR, "metrics_adr.xlsx"))


def _collect_yes_cover_from_files_20250818(analysis_dir: str) -> pd.DataFrame:
    """
    遍历 analysis_dir 下所有文件名包含 '20250818' 的 .xlsx，
    读取 sheet 'details' 的列 ['model','N','llm_answer_yes','cover_valid']，
    计算按 (model, N) 分组的条件概率：
        yes_cover_valid_rate = P(cover_valid=True | llm_answer_yes=True)
    返回列：['model','N','yes_total','valid_yes_total','yes_cover_valid_rate']
    """
    import pandas as pd
    import os

    wanted_cols = ['model', 'N', 'llm_answer_yes', 'cover_valid']
    dfs = []
    for file in os.listdir(analysis_dir):
        if (file.lower().endswith('.xlsx')) and ('20250818' in file) and ("llm_vc_accuracy" in file):
            path = os.path.join(analysis_dir, file)
            try:
                # 尝试只读需要的列，提高稳健性与速度
                df_det = pd.read_excel(path, sheet_name='details', usecols=wanted_cols)
            except Exception:
                # 回退：整表读入再裁剪
                try:
                    df_det = pd.read_excel(path, sheet_name='details')
                except Exception:
                    continue
                if not set(wanted_cols).issubset(df_det.columns):
                    continue
                df_det = df_det[wanted_cols]
            dfs.append(df_det)

    if not dfs:
        return pd.DataFrame(columns=['model','N','yes_total','valid_yes_total','yes_cover_valid_rate'])

    big = pd.concat(dfs, ignore_index=True)

    # 规范布尔
    yes = big['llm_answer_yes']
    if yes.dtype != bool:
        yes = yes.astype(str).str.lower().isin(['true', '1', 'yes', '1.0', 'TRUE'])
    cov = big['cover_valid']
    if cov.dtype != bool:
        cov = cov.astype(str).str.lower().isin(['true', '1', 'yes', '1.0', 'TRUE'])

    big = big.assign(llm_answer_yes_bool=yes, cover_valid_bool=cov)

    # 只在 llm_answer_yes=True 的样本上统计 cover_valid=True 的比例
    sub = big[big['llm_answer_yes_bool'] == True].copy()
    if sub.empty:
        return pd.DataFrame(columns=['model','N','yes_total','valid_yes_total','yes_cover_valid_rate'])

    # 分组聚合
    grouped = (sub
               .groupby(['model', 'N'], dropna=False)
               .agg(yes_total=('llm_answer_yes_bool', 'size'),
                    valid_yes_total=('cover_valid_bool', 'sum'))
               .reset_index())
    grouped['yes_cover_valid_rate'] = grouped.apply(
        lambda r: (r['valid_yes_total'] / r['yes_total']) if r['yes_total'] > 0 else 0.0, axis=1
    )
    return grouped


def merge_yes_rate_with_adr_from_20250818(
        analysis_dir: str = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis',
        metrics_filename: str = 'metrics_adr.xlsx',
        output_filename: str = 'metrics_adr_with_yes_cover_valid_rate_20250818.xlsx',
        force_recompute_adr: bool = False
    ) -> str:
    """
    1) 从所有“文件名包含 20250818”的 xlsx 读取 details -> 只取 llm_answer_yes、cover_valid 做统计，
       得到每个 (model, N) 的 yes_cover_valid_rate。
    2) 读取 metrics_adr.xlsx（若不存在且 force_recompute_adr=True 则先跑 get_all_metrics()）。
    3) 按 (model, N) 外连接合并，并保存为 output_filename。
    """
    import os
    import pandas as pd

    os.makedirs(analysis_dir, exist_ok=True)
    metrics_path = os.path.join(analysis_dir, metrics_filename)

    # 如需，先生成 ADR 文件
    if (force_recompute_adr or (not os.path.exists(metrics_path))) and ('get_all_metrics' in globals()):
        try:
            get_all_metrics()
        except Exception as e:
            print("调用 get_all_metrics() 失败：", e)

    # 读取 ADR 汇总（可能不存在）
    if os.path.exists(metrics_path):
        try:
            df_adr = pd.read_excel(metrics_path)
        except Exception:
            df_adr = pd.DataFrame()
    else:
        df_adr = pd.DataFrame()

    # 读取并计算 20250818 的条件概率
    df_yes = _collect_yes_cover_from_files_20250818(analysis_dir)

    # 统一类型，防止合并出错
    if not df_adr.empty:
        if 'N' in df_adr.columns:
            df_adr['N'] = pd.to_numeric(df_adr['N'], errors='coerce')
        if 'model' in df_adr.columns:
            df_adr['model'] = df_adr['model'].astype(str)
    if not df_yes.empty:
        df_yes['N'] = pd.to_numeric(df_yes['N'], errors='coerce')
        df_yes['model'] = df_yes['model'].astype(str)

    merged = df_yes if df_adr.empty else pd.merge(df_adr, df_yes, on=['model','N'], how='outer')
    out_path = os.path.join(analysis_dir, output_filename)
    merged.to_excel(out_path, index=False)
    print("已生成：", out_path)
    return out_path



if __name__ == "__main__":
    # main()
    # exit()

    # get_all_metrics()
    merge_yes_rate_with_adr_from_20250818()



    # merge_yes_rate_with_adr(force_recompute_adr=True)

