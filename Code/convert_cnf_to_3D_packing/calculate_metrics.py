import json
import os
import pandas as pd
import re



def block_to_dfs(name, block):
    acc_df = pd.DataFrame([{"metric": name, "accuracy": block.get("acc")}])
    conf = block.get("confusion", {}) or {}
    confusion_df = pd.DataFrame([{"metric": name, **conf}])
    sat = block.get("SAT", {}) or {}
    unsat = block.get("UNSAT", {}) or {}
    perclass_df = pd.DataFrame([
        {"metric": name, "class": "SAT", **sat},
        {"metric": name, "class": "UNSAT", **unsat},
    ])
    return acc_df, confusion_df, perclass_df

def add_model_n_columns(dfs: list, model_name, n_value):
    """
    给一组 DataFrame 添加 model_name 和 n_value 两列
    """

    updated = []
    for df in dfs:
        df = df.copy()
        df["model_name"] = model_name
        df["n_value"] = n_value
        updated.append(df)
    return updated

def convert_json_to_df(data, model_name, n_value):
    # Build DataFrames
    overview_df = pd.DataFrame([{"model": data.get("model",""), "num_samples": data.get("num_samples",0)}])
    acc_yes, conf_yes, per_yes = block_to_dfs("pred_llm_yes", data["pred_llm_yes"])
    acc_asg, conf_asg, per_asg = block_to_dfs("pred_assignment_verified", data["pred_assignment_verified"])

    accuracy_df = pd.concat([acc_yes, acc_asg], ignore_index=True)
    confusion_df = pd.concat([conf_yes, conf_asg], ignore_index=True)
    perclass_df = pd.concat([per_yes, per_asg], ignore_index=True)

    dfs = [overview_df, accuracy_df, confusion_df, perclass_df]
    dfs = add_model_n_columns(dfs, model_name, n_value)
    dfs = normalize_dfs(dfs, model_name, n_value)
    return dfs[0], dfs[1], dfs[2], dfs[3]


def normalize_dfs(dfs: list, model_name: str, n_value: int):
    """
    处理多个 DataFrame:
      - 如果存在 'metric' 列，则重命名为 'perspective'
      - 添加 'model_name', 'n_value' 两列，并放到最前
    """
    updated = []
    for df in dfs:
        df = df.copy()

        # 重命名 metric -> perspective
        if "metric" in df.columns:
            df = df.rename(columns={"metric": "perspective"})

        # 添加 model_name, n_value
        df["model_name"] = model_name
        df["n_value"] = n_value

        # 重新排列列顺序
        cols = ["model_name", "n_value"] + [c for c in df.columns if c not in ["model_name", "n_value"]]
        df = df[cols]

        updated.append(df)
    return updated
def parse_model_and_n(filename: str):
    """
    从文件名中提取模型名字和 N 值。
    支持文件名格式:
        <model>_N<number>_summary_metrics.xlsx
        <model>_N<number>_summary_metrics.json
    """
    base = os.path.basename(filename)  # 去掉路径，只取文件名
    m = re.match(r"(.+)_N(\d+)_summary_metrics\.(json|xlsx)", base, re.IGNORECASE)
    if not m:
        raise ValueError(f"文件名格式不匹配: {filename}")
    model_name = m.group(1)
    n_value = int(m.group(2))
    return model_name, n_value


def sort_df(df):
    df["n_value"] = pd.to_numeric(df["n_value"], errors="coerce")
    df = df.sort_values(by=["model_name", "n_value"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return df


# ===== 用法示例 =====
# 假设你的输出目录是 ["out_modelA", "out_modelB", "out_modelC"]
# 每个目录里有 summary_metrics.xlsx
# merged_path = merge_all_summary_excels(["out_modelA", "out_modelB", "out_modelC"])
if __name__ == '__main__':
    model_list = ['o1' ,'gpt-3.5-turbo-0125' ,'chatgpt-4o-latest', 'gpt-4.1' ,'o3-mini' ]  # ,'gpt-5' # or anything you are testing
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/cnf_to_3D_packing'

    excel_dir = []
    all_model_all_N_PerSample = []
    overview_df_all = []
    accuracy_df_all = []
    confusion_df_all = []
    perclass_df_all = []

    for dir in os.listdir(output_dir_root):
        dir_path = os.path.join(output_dir_root, dir)
        if not os.path.isdir(dir_path) or "analysis" in dir:
            continue

        for file in os.listdir(dir_path):

            file_path = os.path.join(dir_path, file)
            if ".xlsx" in file_path and "summary_metrics" in file_path:
                model_name, n_value = parse_model_and_n(file)
                df = pd.read_excel(file_path, 'PerSample')
                print(df)
                df['N_meta'] = n_value
                all_model_all_N_PerSample.append(df)
                excel_dir.append(dir_path)
            if '.json' in file_path:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    print("file_path", file_path)

                    if "pred_llm_yes" in json_data:
                        model_name, n_value = parse_model_and_n(file)
                        overview_df, accuracy_df, confusion_df, perclass_df =convert_json_to_df(json_data, model_name, n_value)
                        overview_df_all.append(overview_df)
                        accuracy_df_all.append(accuracy_df)
                        confusion_df_all.append(confusion_df)
                        perclass_df_all.append(perclass_df)

    out_path = os.path.join(output_dir_root, 'analysis', "summary_metrics.xlsx")
    overview_df_all = pd.concat(overview_df_all)
    accuracy_df_all = pd.concat(accuracy_df_all)
    confusion_df_all = pd.concat(confusion_df_all)
    perclass_df_all = pd.concat(perclass_df_all)
    all_model_all_N_PerSample = pd.concat(all_model_all_N_PerSample)
    overview_df_all = sort_df(overview_df_all)
    accuracy_df_all = sort_df(accuracy_df_all)
    confusion_df_all = sort_df(confusion_df_all)
    perclass_df_all = sort_df(perclass_df_all)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        overview_df_all.to_excel(writer, sheet_name="Overview", index=False)
        accuracy_df_all.to_excel(writer, sheet_name="Accuracy", index=False)
        confusion_df_all.to_excel(writer, sheet_name="Confusion", index=False)
        perclass_df_all.to_excel(writer, sheet_name="PerClass", index=False)
        all_model_all_N_PerSample.to_excel(writer, sheet_name="PerSample_all_model_all_N", index=False)
