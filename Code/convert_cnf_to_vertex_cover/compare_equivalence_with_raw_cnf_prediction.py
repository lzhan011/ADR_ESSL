import copy
import os

import pandas as pd

from convert_cnf_to_vertex_cover_method_1 import  *
from convert_cnf_to_vertex_cover.calculate_accuracy_fixed_pairs import read_fixed_pairs_results
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
        print("fname:", fname)
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




def load_vector_cover_result(ANALYSIS_DIR, MODEL_LIST, INPUT_CNF_ROOT, OUTPUT_ROOT):
    N_LIST = [5, 8, 10, 25, 50, 60]
    # N_LIST = [50]
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    all_rows = []
    all_rows_all_models = {}
    for model in MODEL_LIST:
        print("model:", model)
        for N in N_LIST:
            rows = evaluate(INPUT_CNF_ROOT, OUTPUT_ROOT, model, N)
            all_rows.extend(rows)
            all_rows_all_models[model] = all_rows



def read_vertex_cover_result(ANALYSIS_DIR, N_list):


    file_dict = {}
    for file in os.listdir(ANALYSIS_DIR):
        file_name = copy.deepcopy(file)
        if "llm_vc_accuracy_20250818" in file:
            file = file.replace(".xlsx", "")
            file_N = file.split("_N_")[1]
            file_dict[file_N] = file_name



    print("file_dict:",file_dict)

    all_model_all_N_res = {}
    for N in N_list:
        file_name = file_dict[str(N)]
        file_path = os.path.join(ANALYSIS_DIR, file_name)
        print(file_path)
        cv_df = pd.read_excel(file_path,sheet_name="details")
        print(cv_df.columns.values)
        print(cv_df.shape)

        all_model_all_N_res[N] = cv_df

    return all_model_all_N_res


def calculate_after_match_metrics(all_file_all_model_all_N_vc_and_cnf_prediction, Output_Dir):
    import pandas as pd

    df = all_file_all_model_all_N_vc_and_cnf_prediction.copy()

    bool_cols = [
        "cnf_and_VC_llm_answer_yes_have_same_prediction",
        "cnf_and_vc_and_label_are_same",
        "cnf_and_VC_cover_valid_have_same_prediction",
        "cnf_and_VC_llm_answer_yes_and_VC_cover_valid_have_same_prediction",
    ]

    # （可选）保证是可空布尔类型，保留 NaN
    for c in bool_cols:
        df[c] = df[c].astype("boolean")

    def summarize(group):
        out = {}
        for c in bool_cols:
            true_cnt = int(group[c].sum(skipna=True))  # True 的个数
            total = int(group[c].notna().sum())  # 非 NaN 的个数
            false_cnt = total - true_cnt  # False 的个数
            true_ratio = (true_cnt / total) if total else 0.0  # True 占比（忽略 NaN）
            out[(c, "true_count")] = true_cnt
            out[(c, "false_count")] = false_cnt
            out[(c, "true_ratio")] = round(true_ratio, 4)
        return pd.Series(out)

    wide_summary = (
        df.groupby(["model", "N"], dropna=False)
        .apply(summarize)
        .sort_index()
    )

    # 展平成单层列名并重置索引
    wide_summary.columns = [f"{col}_{stat}" for col, stat in wide_summary.columns]
    wide_summary = wide_summary.reset_index()

    print(wide_summary.head())

    # 如需保存
    wide_summary.to_excel(os.path.join(Output_Dir, "summary_by_model_N_wide.xlsx"), index=False)


def match_fixed_pairs_and_vertex_cover(fixed_pairs_all_model_res, vertex_cover_all_model_all_N_res, model_list, N_list, Output_Dir):

    all_file_all_model_all_N_vc_and_cnf_prediction = []

    for model in model_list:
        print("model:", fixed_pairs_all_model_res[model])
        pairs_res_one_model = fixed_pairs_all_model_res[model]

        for N_tmp in N_list:
            pairs_res_one_model_one_N = pairs_res_one_model[pairs_res_one_model['N'] == N_tmp]
            all_file_one_model_one_N_vc_and_cnf_prediction = []
            for idx, row in pairs_res_one_model_one_N.iterrows():
                pairs_N = int((row['N']))
                pairs_dir = row['dir']
                pairs_file_name = row['file_name']
                if "txt" in pairs_file_name:
                    pairs_file_name = pairs_file_name.replace(".txt", ".cnf")
                pairs_label_IS_SAT = row['label_IS_SAT']
                pairs_prediction_IS_SAT = row['prediction_IS_SAT']

                if int(pairs_N) in vertex_cover_all_model_all_N_res:
                    one_N_all_model_res = vertex_cover_all_model_all_N_res[int(pairs_N)]
                    one_N_one_model_res = one_N_all_model_res[one_N_all_model_res['model'] == model]
                    one_N_one_model_one_file_res = one_N_one_model_res[one_N_one_model_res['instance'] == pairs_file_name ]
                    for vc_index, vc_row in one_N_one_model_one_file_res.iterrows():
                        VC_model = vc_row['model']
                        VC_N = vc_row['N']
                        VC_instance = vc_row['instance']
                        VC_k = vc_row['k']
                        VC_gt_exists = vc_row['gt_exists']
                        VC_gt_from_name = vc_row['gt_from_name']
                        VC_llm_parsed = vc_row['llm_parsed']
                        VC_llm_answer_yes = vc_row['llm_answer_yes']
                        VC_llm_cover_size = vc_row['llm_cover_size']
                        VC_cover_valid = vc_row['cover_valid']
                        VC_label_correct_algo = vc_row['label_correct_algo']
                        VC_label_correct_name = vc_row['label_correct_name']
                        VC_validated_correct = vc_row['validated_correct']


                        if pairs_prediction_IS_SAT == VC_llm_answer_yes:
                            cnf_and_VC_llm_answer_yes_have_same_prediction = True
                            if VC_llm_answer_yes == pairs_label_IS_SAT:
                                cnf_and_vc_and_label_are_same = True
                            else:
                                cnf_and_vc_and_label_are_same = False
                        else:
                            cnf_and_VC_llm_answer_yes_have_same_prediction = False
                            cnf_and_vc_and_label_are_same = False


                        if pairs_prediction_IS_SAT == VC_cover_valid:
                            cnf_and_VC_cover_valid_have_same_prediction = True
                        else:
                            cnf_and_VC_cover_valid_have_same_prediction = False


                        if pairs_prediction_IS_SAT == VC_llm_answer_yes and VC_llm_answer_yes == VC_cover_valid:
                            cnf_and_VC_llm_answer_yes_and_VC_cover_valid_have_same_prediction = True
                        else:
                            cnf_and_VC_llm_answer_yes_and_VC_cover_valid_have_same_prediction = False





                        all_file_one_model_one_N_vc_and_cnf_prediction.append({
                            "model": model,
                            "N": pairs_N,
                           "file_name":pairs_file_name,
                           "cnf_dir":pairs_dir,
                           "cnf_label_IS_SAT":pairs_label_IS_SAT,
                           "cnf_prediction_IS_SAT":pairs_prediction_IS_SAT,
                            "VC_k":VC_k,
                            "VC_N":VC_N,
                            "VC_instance":VC_instance,
                            "VC_gt_exists":VC_gt_exists,
                            "VC_gt_from_name":VC_gt_from_name,
                            "VC_llm_parsed":VC_llm_parsed,
                            "VC_llm_answer_yes":VC_llm_answer_yes,
                            "VC_llm_cover_size":VC_llm_cover_size,
                            "VC_cover_valid":VC_cover_valid,
                            "VC_label_correct_algo":VC_label_correct_algo,
                            "VC_label_correct_name":VC_label_correct_name,
                            "VC_validated_correct":VC_validated_correct,
                            "cnf_and_VC_llm_answer_yes_have_same_prediction":cnf_and_VC_llm_answer_yes_have_same_prediction,
                            "cnf_and_vc_and_label_are_same":cnf_and_vc_and_label_are_same,
                            "cnf_and_VC_cover_valid_have_same_prediction":cnf_and_VC_cover_valid_have_same_prediction,
                            "cnf_and_VC_llm_answer_yes_and_VC_cover_valid_have_same_prediction":cnf_and_VC_llm_answer_yes_and_VC_cover_valid_have_same_prediction,
                        })
            all_file_one_model_one_N_vc_and_cnf_prediction = pd.DataFrame(all_file_one_model_one_N_vc_and_cnf_prediction)
            all_file_one_model_one_N_vc_and_cnf_prediction.to_excel(
                os.path.join(Output_Dir, f"all_file_one_model_{model}_one_N_{str(pairs_N)}_vc_and_cnf_prediction.xlsx"))
            all_file_all_model_all_N_vc_and_cnf_prediction.append(all_file_one_model_one_N_vc_and_cnf_prediction)

    all_file_all_model_all_N_vc_and_cnf_prediction = pd.concat(all_file_all_model_all_N_vc_and_cnf_prediction)
    all_file_all_model_all_N_vc_and_cnf_prediction.to_excel(os.path.join(Output_Dir, 'all_file_all_model_all_N_vc_and_cnf_prediction.xlsx'))
    calculate_after_match_metrics(all_file_all_model_all_N_vc_and_cnf_prediction, Output_Dir)





if __name__ == '__main__':
    vertex_cover_dir = ''
    # ---------------------- 配置 ----------------------
    INPUT_CNF_ROOT = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    OUTPUT_ROOT = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/vertex_cover_graph'
    ANALYSIS_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis'
    cnf_and_vertex_cover_equivalence_dir = os.path.join(ANALYSIS_DIR, 'cnf_and_vertex_cover_equivalence')
    MODEL_LIST = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', 'gpt-4o',
                  'o3-mini', 'deepseek-reasoner', 'o1', 'gpt-5']
    N_list = [5, 8, 10, 25]
    # load_vector_cover_result(ANALYSIS_DIR, MODEL_LIST, INPUT_CNF_ROOT, OUTPUT_ROOT)

    fixed_pairs_root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'

    fixed_pairs_model_list = ['deepseek-reasoner' , 'o1', 'o3-mini', 'gpt-5'
              , 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', 'gpt-4o',
                   ]
    fixed_pairs_all_model_res = read_fixed_pairs_results(fixed_pairs_model_list, fixed_pairs_root_dir)
    vertex_cover_all_model_all_N_res = read_vertex_cover_result(ANALYSIS_DIR, N_list)

    match_fixed_pairs_and_vertex_cover(fixed_pairs_all_model_res, vertex_cover_all_model_all_N_res, fixed_pairs_model_list, N_list, cnf_and_vertex_cover_equivalence_dir)



