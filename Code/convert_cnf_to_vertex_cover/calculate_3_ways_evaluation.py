import os


def build_pair_predictions_from_details_20250818(
        analysis_dir: str = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis',
        output_filename: str = 'pair_predictions_all_models_20250818.xlsx'
    ) -> str:
    """
    从 analysis_dir 下所有文件名包含 '20250818' 且包含 'llm_vc_accuracy' 的 .xlsx 的 sheet='details'
    读取列: ['model','N','instance','llm_answer_yes','cover_valid']，
    生成每一对 (orig vs _RC2_fixed) 的汇总行，输出列:
      ['model','pair_base_name','N',
       'original_file','original_prediction',
       'fixed_file','fixed_prediction',
       'fixed_cover_valid'].

    注意：
    - original_prediction/fixed_prediction 分别通过对 original_file/fixed_file 在输入明细中的
      llm_answer_yes 进行“按文件名精确查表”得到。
    - fixed_cover_valid 通过 fixed_file 在输入明细中“按文件名精确查表”得到 cover_valid。
    - 仅保留完整 pair（orig 与 fixed 都存在）。
    """
    import os
    import pandas as pd

    def _safe_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        sl = s.astype(str).str.strip().str.lower()
        return sl.isin(['true', '1', '1.0', 'yes', 'y', 't'])

    def _pair_base(inst: str) -> str:
        stem = os.path.splitext(str(inst))[0]
        return stem[:-len("_RC2_fixed")] if stem.endswith("_RC2_fixed") else stem

    def _is_fixed(inst: str) -> bool:
        return os.path.splitext(str(inst))[0].endswith("_RC2_fixed")

    # 1) 收集所有 details
    wanted_cols = ['model','N','instance','llm_answer_yes','cover_valid']
    dfs = []
    for fn in os.listdir(analysis_dir):
        if (fn.lower().endswith('.xlsx')) and ('20250818' in fn) and ('llm_vc_accuracy' in fn):
            path = os.path.join(analysis_dir, fn)
            try:
                df = pd.read_excel(path, sheet_name='details', usecols=wanted_cols)
            except Exception:
                try:
                    df = pd.read_excel(path, sheet_name='details')
                except Exception:
                    continue
                if not set(wanted_cols).issubset(df.columns):
                    continue
                df = df[wanted_cols]
            dfs.append(df)

    if not dfs:
        print("[pairs-20250818] 未找到可用的 details 数据源。")
        return ""

    big = pd.concat(dfs, ignore_index=True)

    # 2) 规范类型
    big['model'] = big['model'].astype(str)
    big['N'] = pd.to_numeric(big['N'], errors='coerce')
    big = big.dropna(subset=['N', 'instance']).copy()
    big['N'] = big['N'].astype(int)

    # 规范布尔
    big['llm_answer_yes'] = _safe_bool_series(big['llm_answer_yes'])
    big['cover_valid'] = _safe_bool_series(big['cover_valid'])

    # 辅助列
    big['pair_base_name'] = big['instance'].apply(_pair_base)
    big['__is_fixed__']   = big['instance'].apply(_is_fixed)

    # 同一 (model, N, instance) 若出现多次，仅保留最后一条
    big = big.sort_index().drop_duplicates(subset=['model','N','instance'], keep='last')

    # 3) 构建查表映射：
    #    (model, N, instance) -> llm_answer_yes / cover_valid
    key_cols = ['model', 'N', 'instance']
    pred_map = {(r['model'], r['N'], r['instance']): bool(r['llm_answer_yes'])
                for _, r in big[key_cols + ['llm_answer_yes']].iterrows()}
    cover_map = {(r['model'], r['N'], r['instance']): bool(r['cover_valid'])
                 for _, r in big[key_cols + ['cover_valid']].iterrows()}

    # 4) 生成 pair（拿到 orig/fixed 的文件名）
    orig = (big[~big['__is_fixed__']][['model','N','pair_base_name','instance']]
            .rename(columns={'instance':'original_file'}))
    fixed = (big[ big['__is_fixed__']][['model','N','pair_base_name','instance']]
            .rename(columns={'instance':'fixed_file'}))

    pairs = pd.merge(orig, fixed, on=['model','N','pair_base_name'], how='inner')

    # 5) 查表填充 original_prediction / fixed_prediction / fixed_cover_valid
    def _lookup_pred(row, which: str):
        file_key = row[f'{which}_file']
        return pred_map.get((row['model'], row['N'], file_key), None)

    def _lookup_cover(row):
        return cover_map.get((row['model'], row['N'], row['fixed_file']), None)

    pairs['original_prediction'] = pairs.apply(lambda r: _lookup_pred(r, 'original'), axis=1)
    pairs['fixed_prediction']    = pairs.apply(lambda r: _lookup_pred(r, 'fixed'), axis=1)
    pairs['fixed_cover_valid']   = pairs.apply(lambda r: _lookup_cover(r), axis=1)

    # 6) 排序与列次序
    pairs = pairs[['model','pair_base_name','N',
                   'original_file','original_prediction',
                   'fixed_file','fixed_prediction',
                   'fixed_cover_valid']].sort_values(['model','N','pair_base_name'])

    # 7) 保存
    out_dir = os.path.join(analysis_dir, 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    pairs.to_excel(out_path, index=False)
    print(f"[pairs-20250818] 已保存：{out_path}（{len(pairs)} 对）")
    return pairs, analysis_dir


import pandas as pd
import os
import pandas as pd

def compute_success_rate_with_cover(pairs: pd.DataFrame, analysis_dir) -> pd.DataFrame:
    """
    统计每个 (model, N) 下:
      - success_count: original_prediction==False & fixed_prediction==True & fixed_cover_valid==True
      - origF_fixT_count: original_prediction==False & fixed_prediction==True
      - fixed_cover_valid_count: fixed_cover_valid==True

    返回 DataFrame 列：
      ['model','N','total_pairs','success_count','success_rate',
       'origF_fixT_count','origF_fixT_rate',
       'fixed_cover_valid_count','fixed_cover_valid_rate']
    """
    df = pairs.copy()

    # 统一布尔类型
    def _to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y", "t"]
        )

    for col in ["original_prediction", "fixed_prediction", "fixed_cover_valid"]:
        df[col] = _to_bool(df[col])

    # 条件列
    df["success"] = (~df["original_prediction"]) & df["fixed_prediction"] & df["fixed_cover_valid"]
    df["origF_fixT"] = (~df["original_prediction"]) & df["fixed_prediction"]
    df["fcv_true"] = df["fixed_cover_valid"]

    # 分组统计
    grouped = (
        df.groupby(["model", "N"], dropna=False)
          .agg(
              total_pairs=("pair_base_name", "size"),
              success_count=("success", "sum"),
              origF_fixT_count=("origF_fixT", "sum"),
              fixed_cover_valid_count=("fcv_true", "sum"),
          )
          .reset_index()
    )

    # 各比例
    grouped["success_rate"] = grouped["success_count"] / grouped["total_pairs"]
    grouped["origF_fixT_rate"] = grouped["origF_fixT_count"] / grouped["total_pairs"]
    grouped["fixed_cover_valid_rate"] = grouped["fixed_cover_valid_count"] / grouped["total_pairs"]

    out_path = os.path.join(analysis_dir, 'three_ways_evaluation', 'three_ways_evaluation.xlsx')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grouped.to_excel(out_path, index=False)
    print(f"[compute_success_rate_with_cover] 已保存: {out_path}")
    return grouped




if __name__ == "__main__":

    # 新增：基于 20250818 的 details 汇总生成 pair 预测结果（英文列名）
    pairs, analysis_dir = build_pair_predictions_from_details_20250818()

    compute_success_rate_with_cover(pairs, analysis_dir)
