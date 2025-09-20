import os, re
import glob
from pathlib import Path
import pandas as pd
import json
import glob
from pathlib import Path

# === Build pairs from PerSample (do not modify existing code above) ===
def _pair_base_from_filename(fn: str) -> str:
    """去掉扩展名，再去掉末尾的 _RC2_fixed 作为配对基名"""
    stem = os.path.splitext(str(fn))[0]
    return stem[:-len("_RC2_fixed")] if stem.endswith("_RC2_fixed") else stem

def _is_fixed_filename(fn: str) -> bool:
    """判断是否为 fix 版本"""
    return os.path.splitext(str(fn))[0].endswith("_RC2_fixed")

def _to_bool_series(s: pd.Series) -> pd.Series:
    """把 True/False/1/0/yes/no 字样统一为布尔"""
    if s.dtype == bool:
        return s
    sl = s.astype(str).str.strip().str.lower()
    return sl.isin(["true", "1", '1.0', 'TRUE', 'True', "yes", "y", "t"])

def build_pairs_from_per_sample(per_sample_df: pd.DataFrame,
                                out_root: str,
                                out_filename: str = "pairs_from_per_sample.xlsx") -> str:
    """
    从 all_model_all_N_PerSample（列：
      ['filename','model','N_meta','alpha_meta','ground_truth',
       'pred_llm_yes','pred_assignment_verified','source_answer_path']）
    中配对生成：
      ['model','pair_base_name','N',
       'original_file','original_prediction',
       'fixed_file','fixed_prediction',
       'fixed_assignment_verified']

    说明：
    - original_prediction / fixed_prediction 分别通过查表
      (model, N, filename) -> pred_llm_yes
    - fixed_assignment_verified 通过查表
      (model, N, fixed_file) -> pred_assignment_verified
    """
    if per_sample_df is None or per_sample_df.empty:
        print("[pairs-from-PerSample] 输入 DataFrame 为空，跳过。")
        return ""

    df = per_sample_df.copy()

    # 只保留必要列并检查
    need_cols = ['filename', 'model', 'N_meta', 'pred_llm_yes', 'pred_assignment_verified']
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"[pairs-from-PerSample] 缺少必要列：{missing}")

    # 规范类型
    df = df[need_cols].copy()
    df['model'] = df['model'].astype(str)
    df['N'] = pd.to_numeric(df['N_meta'], errors='coerce')
    df = df.dropna(subset=['N', 'filename']).copy()
    df['N'] = df['N'].astype(int)

    # 统一布尔
    def _to_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        sl = s.astype(str).str.strip().str.lower()
        return sl.isin(["true", 'sat', 'SAT', '1.0',"1", "yes", "y", "t"])

    df['pred_llm_yes'] = _to_bool_series(df['pred_llm_yes'])
    df['pred_assignment_verified'] = _to_bool_series(df['pred_assignment_verified'])

    # 构建查表字典
    pred_yes_lookup = {(row['model'], row['N'], str(row['filename'])): bool(row['pred_llm_yes'])
                       for _, row in df[['model', 'N', 'filename', 'pred_llm_yes']].iterrows()}
    assign_verified_lookup = {(row['model'], row['N'], str(row['filename'])): bool(row['pred_assignment_verified'])
                              for _, row in df[['model', 'N', 'filename', 'pred_assignment_verified']].iterrows()}

    # 基名与 fixed 判定
    def _pair_base_from_filename(fn: str) -> str:
        stem = os.path.splitext(str(fn))[0]
        return stem[:-len("_RC2_fixed")] if stem.endswith("_RC2_fixed") else stem

    def _is_fixed_filename(fn: str) -> bool:
        return os.path.splitext(str(fn))[0].endswith("_RC2_fixed")

    df['pair_base_name'] = df['filename'].apply(_pair_base_from_filename)
    df['__is_fixed__']   = df['filename'].apply(_is_fixed_filename)

    # 去重（同一 (model, N, filename) 只保留最后一条）
    df = df.sort_index().drop_duplicates(subset=['model', 'N', 'filename'], keep='last')

    # 拆分 orig / fixed（先带文件名，预测稍后查表填充）
    orig = (df[~df['__is_fixed__']][['model', 'N', 'pair_base_name', 'filename']]
            .rename(columns={'filename': 'original_file'}))
    fixed = (df[df['__is_fixed__']][['model', 'N', 'pair_base_name', 'filename']]
             .rename(columns={'filename': 'fixed_file'}))

    # 只保留完整 pair
    pairs = pd.merge(orig, fixed, on=['model', 'N', 'pair_base_name'], how='inner')

    # 查表填充
    def _lookup_yes(m, n, fn):
        return pred_yes_lookup.get((m, n, str(fn)), None)

    def _lookup_assign(m, n, fn):
        return assign_verified_lookup.get((m, n, str(fn)), None)

    pairs['original_prediction'] = pairs.apply(
        lambda r: _lookup_yes(r['model'], r['N'], r['original_file']), axis=1
    )
    pairs['fixed_prediction'] = pairs.apply(
        lambda r: _lookup_yes(r['model'], r['N'], r['fixed_file']), axis=1
    )
    pairs['fixed_assignment_verified'] = pairs.apply(
        lambda r: _lookup_assign(r['model'], r['N'], r['fixed_file']), axis=1
    )

    # 排序与列顺序
    pairs = pairs[['model', 'pair_base_name', 'N',
                   'original_file', 'original_prediction',
                   'fixed_file', 'fixed_prediction',
                   'fixed_assignment_verified']].sort_values(['model', 'N', 'pair_base_name'])

    # 写出
    out_dir = os.path.join(out_root, 'analysis' , 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    pairs.to_excel(out_path, index=False)
    print(f"[pairs-from-PerSample] 已保存：{out_path}（{len(pairs)} 对）")
    return pairs






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


import os
import pandas as pd

def compute_success_rate(pairs: pd.DataFrame, output_dir_root) -> pd.DataFrame:
    """
    对 pairs 表格，计算每个 (model, N) 下：
      - success_count: original_prediction==False & fixed_prediction==True & fixed_assignment_verified==True 的数量
      - origF_fixT_count: original_prediction==False & fixed_prediction==True 的数量
      - fixed_assignment_verified_count: fixed_assignment_verified==True 的数量
      - success_rate = success_count / total_pairs
    返回列：
      ['model','N','total_pairs','success_count','origF_fixT_count',
       'fixed_assignment_verified_count','success_rate']
    """
    df = pairs.copy()

    # 统一布尔类型
    def _to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

    for col in ["original_prediction", "fixed_prediction", "fixed_assignment_verified"]:
        df[col] = _to_bool(df[col])

    # 条件列
    df["success"] = (~df["original_prediction"]) & (df["fixed_prediction"]) & (df["fixed_assignment_verified"])
    df["origF_fixT"] = (~df["original_prediction"]) & (df["fixed_prediction"])
    df["fav_true"] = df["fixed_assignment_verified"]

    # 分组统计
    grouped = (
        df.groupby(["model", "N"], dropna=False)
          .agg(
              total_pairs=("pair_base_name", "size"),
              success_count=("success", "sum"),
              origF_fixT_count=("origF_fixT", "sum"),
              fixed_assignment_verified_count=("fav_true", "sum"),
          )
          .reset_index()
    )
    grouped["success_rate"] = grouped["success_count"] / grouped["total_pairs"]
    grouped["origF_fixT_rate"] = grouped["origF_fixT_count"] / grouped["total_pairs"]
    grouped["fixed_assignment_verified_rate"] = grouped["fixed_assignment_verified_count"] / grouped["total_pairs"]

    out_dir = os.path.join(output_dir_root, 'analysis', 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)
    grouped.to_excel(os.path.join(out_dir, "three_ways_success_rate.xlsx"), index=False)
    return grouped


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

    # 调用：在你现有写 summary_metrics.xlsx 之后追加调用即可
    pairs = build_pairs_from_per_sample(all_model_all_N_PerSample, output_dir_root)

    compute_success_rate(pairs, output_dir_root)