import os, re
import glob
from pathlib import Path
import pandas as pd

import glob
from pathlib import Path

def _strip_ext(name: str) -> str:
    # 去除最后的扩展名（.txt/.cnf 等），更稳健
    return re.sub(r'\.[^.]+$', '', str(name))

def _normalize_col(s: str) -> str:
    # 规范化列名：去空白、转小写、去非字母数字与下划线
    s = str(s).strip().lower()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^a-z0-9_]+', '', s)
    return s

def _auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将各种可能变体列名映射到标准名：
       - file_name
       - prediction_is_sat
       - n
    """
    original_cols = list(df.columns)
    mapping = {}
    for c in original_cols:
        norm = _normalize_col(c)

        # file_name 的候选
        if norm in {
            'file_name', 'filename', 'file', 'fname', 'name', 'file_path', 'filepath', 'path'
        }:
            mapping[c] = 'file_name'
            continue

        # prediction_IS_SAT 的候选
        if norm in {
            'prediction_is_sat', 'predictionissat', 'prediction', 'pred', 'is_sat', 'issat'
        }:
            mapping[c] = 'prediction_IS_SAT'
            continue

        # N 的候选
        if norm in {'n'}:
            mapping[c] = 'N'
            continue

        # 其他列保持不变
        # （包括 dir / label_IS_SAT 等）
    if mapping:
        df = df.rename(columns=mapping)

    # 二次兜底：如果还是没有 file_name / prediction_IS_SAT，就尝试按关键词搜
    cols_lower = {c.lower(): c for c in df.columns}

    if 'file_name' not in df.columns:
        # 找到包含 "file" 或 "path" 的列
        candidate = None
        for c in df.columns:
            lc = c.lower()
            if 'file' in lc or 'path' in lc or 'name' in lc:
                candidate = c
                break
        if candidate is not None:
            df = df.rename(columns={candidate: 'file_name'})

    if 'prediction_IS_SAT' not in df.columns:
        candidate = None
        for c in df.columns:
            lc = c.lower()
            if 'predict' in lc or 'is_sat' in lc or 'issat' in lc:
                candidate = c
                break
        if candidate is not None:
            df = df.rename(columns={candidate: 'prediction_IS_SAT'})

    if 'N' not in df.columns:
        # 有些情况下 N 被写成 n 或者作为字符串列
        if 'n' in cols_lower:
            df = df.rename(columns={cols_lower['n']: 'N'})

    # 最终检查 + 抛出更友好的错误
    missing = [x for x in ['file_name', 'prediction_IS_SAT', 'N'] if x not in df.columns]
    if missing:
        raise KeyError(
            f"期望列缺失：{missing}；当前列为：{list(df.columns)}。"
            "请检查该模型导出的 *_instances_res_cross_n.xlsx 表头是否异常（空格/中文/大小写/合并单元格）。"
        )
    return df

def build_and_save_pair_predictions_from_instances(root_dir: str):
    """
    Scan all *_instances_res_cross_n.xlsx under root_dir/analysis,
    build per-model pair results, and one combined file.
    Output column names are in English.
    Additionally, join is_satisfied from Assigned_value_satisfied_result_per_file.csv
    by (model/model_name, N, fixed_file/file_name).
    """
    analysis_dir = os.path.join(root_dir, 'analysis')
    out_dir = os.path.join(analysis_dir, 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    excel_paths = glob.glob(os.path.join(analysis_dir, "*_instances_res_cross_n.xlsx"))
    if not excel_paths:
        print(f"[pair-builder] No *_instances_res_cross_n.xlsx found in {analysis_dir}, skip.")
        return

    all_models_rows = []
    for xls_path in excel_paths:
        model_name = Path(xls_path).name.replace("_instances_res_cross_n.xlsx", "")
        try:
            df = pd.read_excel(xls_path)
        except Exception as e:
            print(f"[pair-builder] Read failed: {xls_path} ({e})")
            continue

        try:
            df = _auto_rename_columns(df)
        except KeyError as e:
            print(f"[pair-builder] Missing expected columns, skip {xls_path}\n  Detail: {e}")
            continue

        # Ensure N is int
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df = df.dropna(subset=["N"]).copy()
        df["N"] = df["N"].astype(int)

        # Add fixed flag and basename
        df["__is_fixed__"] = df["file_name"].astype(str).str.contains("RC2_fixed", case=False, regex=False)
        df["__basename__"] = (
            df["file_name"].astype(str)
                .str.replace("_RC2_fixed", "", regex=False)
                .apply(_strip_ext)
        )

        df_min = df[["N", "file_name", "prediction_IS_SAT", "__is_fixed__", "__basename__"]].copy()

        # Separate original vs fixed
        df_orig = df_min[~df_min["__is_fixed__"]].rename(columns={
            "file_name": "original_file",
            "prediction_IS_SAT": "original_prediction"
        })[["N", "__basename__", "original_file", "original_prediction"]]

        df_fix = df_min[df_min["__is_fixed__"]].rename(columns={
            "file_name": "fixed_file",
            "prediction_IS_SAT": "fixed_prediction"
        })[["N", "__basename__", "fixed_file", "fixed_prediction"]]

        # Join pairs
        pairs = pd.merge(df_orig, df_fix, on=["N", "__basename__"], how="inner")

        # Add model and base name
        pairs.insert(0, "model", model_name)
        pairs.insert(1, "pair_base_name", pairs["__basename__"])
        pairs.drop(columns=["__basename__"], inplace=True)

        # Save per-model file
        per_model_out = os.path.join(out_dir, f"{model_name}_pair_predictions.xlsx")
        pairs.to_excel(per_model_out, index=False)
        print(f"[pair-builder] Saved: {per_model_out} ({len(pairs)} pairs)")

        all_models_rows.append(pairs)

    if all_models_rows:
        all_pairs = pd.concat(all_models_rows, ignore_index=True)

        # ==== NEW: merge is_satisfied from per-file CSV ====
        sat_csv = os.path.join(
            analysis_dir, "three_ways_evaluation", "Assigned_value_satisfied_result_per_file.csv"
        )
        if os.path.exists(sat_csv):
            try:
                df_sat = pd.read_csv(sat_csv)
                # 只保留需要的列，并重命名以匹配 key
                needed = ['model_name', 'N', 'file_name', 'is_satisfied']
                missing = [c for c in needed if c not in df_sat.columns]
                if missing:
                    print(f"[pair-builder] WARNING: columns {missing} not found in {sat_csv}; skip merging is_satisfied.")
                else:
                    df_sat = df_sat[needed].copy()
                    # 规范类型，避免合并类型不一致
                    df_sat['model_name'] = df_sat['model_name'].astype(str)
                    df_sat['N'] = pd.to_numeric(df_sat['N'], errors='coerce')
                    df_sat = df_sat.dropna(subset=['N'])
                    df_sat['N'] = df_sat['N'].astype(int)

                    # 防止布尔被当作字符串
                    if df_sat['is_satisfied'].dtype != bool:
                        df_sat['is_satisfied'] = df_sat['is_satisfied'].astype(str).str.strip().str.lower().isin(
                            ['true', '1', 'yes', 'y', 't']
                        )

                    # 为合并准备 key：model==model_name, N==N, fixed_file==file_name
                    df_sat = df_sat.rename(columns={'model_name': 'model', 'file_name': 'fixed_file'})

                    # 合并
                    all_pairs = pd.merge(
                        all_pairs, df_sat[['model', 'N', 'fixed_file', 'is_satisfied']],
                        on=['model', 'N', 'fixed_file'], how='left'
                    )
                    print(f"[pair-builder] Merged is_satisfied from {sat_csv}")
            except Exception as e:
                print(f"[pair-builder] Failed to merge is_satisfied from {sat_csv}: {e}")
        else:
            print(f"[pair-builder] WARNING: per-file CSV not found: {sat_csv}; skip merging is_satisfied.")

        # Save combined file (with is_satisfied if merged)
        all_out = os.path.join(out_dir, "pair_predictions_all_models.xlsx")
        all_pairs.to_excel(all_out, index=False)
        print(f"[pair-builder] Saved combined results: {all_out} ({len(all_pairs)} pairs)")
    else:
        print("[pair-builder] No pair results to combine.")

    return all_pairs, out_dir

import pandas as pd


def compute_success_probability(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    对 all_pairs 计算每个 (model, N) 的统计：
      - success_count: original_prediction==False & fixed_prediction==True & is_satisfied==True 的数量
      - success_probability: success_count / 总数
      - origF_fixT_count: original_prediction==False & fixed_prediction==True 的数量
      - satisfied_count: is_satisfied==True 的数量
    """

    # 确保布尔列统一为 bool
    def _to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y', 't'])

    all_pairs = all_pairs.copy()
    all_pairs['original_prediction'] = _to_bool(all_pairs['original_prediction'])
    all_pairs['fixed_prediction'] = _to_bool(all_pairs['fixed_prediction'])
    all_pairs['is_satisfied'] = _to_bool(all_pairs['is_satisfied'])

    results = []
    for (m, n), g in all_pairs.groupby(['model', 'N']):
        total = len(g)

        # 条件统计
        mask_success = (~g['original_prediction']) & (g['fixed_prediction']) & (g['is_satisfied'])
        mask_origF_fixT = (~g['original_prediction']) & (g['fixed_prediction'])
        mask_satisfied = g['is_satisfied']

        success_count = mask_success.sum()
        origF_fixT_count = mask_origF_fixT.sum()
        satisfied_count = mask_satisfied.sum()

        prob = success_count / total if total > 0 else 0.0

        results.append({
            'model': m,
            'N': n,
            'total_pairs': total,
            'success_count': success_count,
            'origF_fixT_count': origF_fixT_count,
            'satisfied_count': satisfied_count,
            'success_probability': prob
        })
    return pd.DataFrame(results).sort_values(['model', 'N'])


if __name__ == '__main__':
    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'

    # —— 你可以在主流程最后追加这一行来生成结果（不会影响你现有逻辑）——
    all_pairs, out_dir = build_and_save_pair_predictions_from_instances(root_dir)

    df_prob = compute_success_probability(all_pairs)
    print(df_prob)
    # 保存结果
    df_prob.to_excel(os.path.join(out_dir, "three_ways_evaluation_success_probability_by_model_N.xlsx"), index=False)