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

def build_and_save_pair_predictions_from_instances_cn(root_dir: str):
    """
    扫描 root_dir/analysis 下所有 *_instances_res_cross_n.xlsx，
    生成每模型的 pair 结果与一个跨模型汇总表（中文列名版本）。
    关键四列：
      - 文件名不包含fix的文件名
      - 不包含fix文件的预测
      - 文件名包含fix的文件名
      - 包含fix的文件的预测
    另外保留辅助列：model, pair_base_name, N
    """
    analysis_dir = os.path.join(root_dir, 'analysis')
    out_dir = os.path.join(analysis_dir, 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    excel_paths = glob.glob(os.path.join(analysis_dir, "*_instances_res_cross_n.xlsx"))
    if not excel_paths:
        print(f"[pair-builder CN] 未在 {analysis_dir} 找到 *_instances_res_cross_n.xlsx，跳过。")
        return

    all_models_rows = []
    for xls_path in excel_paths:
        model_name = Path(xls_path).name.replace("_instances_res_cross_n.xlsx", "")
        try:
            df = pd.read_excel(xls_path)
        except Exception as e:
            print(f"[pair-builder CN] 读取失败：{xls_path} ({e})")
            continue

        # 规范列名
        try:
            df = _auto_rename_columns(df)
        except KeyError as e:
            print(f"[pair-builder CN] 列名不一致导致跳过：{xls_path}\n  详情：{e}")
            continue

        # N -> int
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df = df.dropna(subset=["N"]).copy()
        df["N"] = df["N"].astype(int)

        # 标记 fixed 与 pair 基名
        df["__is_fixed__"] = df["file_name"].astype(str).str.contains("RC2_fixed", case=False, regex=False)
        df["__basename__"] = (
            df["file_name"].astype(str)
              .str.replace("_RC2_fixed", "", regex=False)
              .apply(_strip_ext)
        )

        # 只保留必要列
        df_min = df[["N", "file_name", "prediction_IS_SAT", "__is_fixed__", "__basename__"]].copy()

        # 拆分：原始 与 fixed
        df_orig = df_min[~df_min["__is_fixed__"]].rename(columns={
            "file_name": "文件名不包含fix的文件名",
            "prediction_IS_SAT": "不包含fix文件的预测"
        })[["N", "__basename__", "文件名不包含fix的文件名", "不包含fix文件的预测"]]

        df_fix = df_min[df_min["__is_fixed__"]].rename(columns={
            "file_name": "文件名包含fix的文件名",
            "prediction_IS_SAT": "包含fix的文件的预测"
        })[["N", "__basename__", "文件名包含fix的文件名", "包含fix的文件的预测"]]

        # 以 (N, 基名) 对齐，仅保留完整 pair
        pairs = pd.merge(df_orig, df_fix, on=["N", "__basename__"], how="inner")

        # 增加模型与基名列
        pairs.insert(0, "model", model_name)
        pairs.insert(1, "pair_base_name", pairs["__basename__"])
        pairs.drop(columns=["__basename__"], inplace=True)

        # 如果你只想要四列，可启用如下两行，覆盖选择列：
        # pairs = pairs[["文件名不包含fix的文件名", "不包含fix文件的预测", "文件名包含fix的文件名", "包含fix的文件的预测"]]

        # 保存每模型文件（中文列名）
        per_model_out = os.path.join(out_dir, f"{model_name}_pair_predictions_CN.xlsx")
        pairs.to_excel(per_model_out, index=False)
        print(f"[pair-builder CN] 已保存：{per_model_out}（{len(pairs)} 对）")

        all_models_rows.append(pairs)

    if all_models_rows:
        all_pairs = pd.concat(all_models_rows, ignore_index=True)
        all_out = os.path.join(out_dir, "pair_predictions_all_models_CN.xlsx")
        all_pairs.to_excel(all_out, index=False)
        print(f"[pair-builder CN] 已保存跨模型汇总：{all_out}（{len(all_pairs)} 对）")
    else:
        print("[pair-builder CN] 没有可汇总的 pair 结果。")


def build_and_save_pair_predictions_from_instances(root_dir: str):
    """
    Scan all *_instances_res_cross_n.xlsx under root_dir/analysis,
    build per-model pair results, and one combined file.
    Output column names are in English.
    Additionally, join 'is_satisfied' from the per-file Excel by (model/model_name, N, fixed_file/file_name).
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

        # ===== NEW: merge is_satisfied from per-file Excel =====
        per_file_excel = "/work/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL/prediction_result/analysis/three_ways_evaluation/All_Model_cnf2_Assigned_value_satisfied_per_file.xlsx"
        if os.path.exists(per_file_excel):
            try:
                df_sat = pd.read_excel(per_file_excel)
                needed = ["model_name", "N", "file_name", "is_satisfied"]
                miss = [c for c in needed if c not in df_sat.columns]
                if miss:
                    print(f"[pair-builder] WARNING: columns {miss} not found in {per_file_excel}; skip merging is_satisfied.")
                else:
                    # 只保留所需列并规范类型
                    df_sat = df_sat[needed].copy()
                    df_sat["model_name"] = df_sat["model_name"].astype(str)
                    df_sat["N"] = pd.to_numeric(df_sat["N"], errors="coerce").astype("Int64")
                    # 统一 is_satisfied 为布尔
                    if df_sat["is_satisfied"].dtype != bool:
                        df_sat["is_satisfied"] = (
                            df_sat["is_satisfied"]
                              .astype(str).str.strip().str.lower()
                              .isin(["true", "TRUE", "1.0", "True", "sat", "1", "yes", "y", "t"])
                        )

                    # 重命名以匹配 all_pairs 的键
                    df_sat = df_sat.rename(columns={
                        "model_name": "model",
                        "file_name": "fixed_file"
                    })
                    # N -> int 并过滤 NA
                    df_sat = df_sat.dropna(subset=["N"]).copy()
                    df_sat["N"] = df_sat["N"].astype(int)

                    # 左连接：按 (model, N, fixed_file)
                    all_pairs = pd.merge(
                        all_pairs,
                        df_sat[["model", "N", "fixed_file", "is_satisfied"]],
                        on=["model", "N", "fixed_file"],
                        how="left"
                    )
                    print(f"[pair-builder] Merged is_satisfied from {per_file_excel}")
            except Exception as e:
                print(f"[pair-builder] Failed to merge is_satisfied: {e}")
        else:
            print(f"[pair-builder] WARNING: per-file Excel not found: {per_file_excel}; skip merging is_satisfied.")

        # Save combined file (with is_satisfied if merged)
        all_out = os.path.join(out_dir, "pair_predictions_all_models.xlsx")
        all_pairs.to_excel(all_out, index=False)
        print(f"[pair-builder] Saved combined results: {all_out} ({len(all_pairs)} pairs)")
    else:
        print("[pair-builder] No pair results to combine.")


    return all_pairs, out_dir



import pandas as pd

def compute_success_rate_with_is_satisfied(all_pairs: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    统计每个 (model, N) 下：
      - total_pairs: 总对数
      - success_count: original_prediction==False & fixed_prediction==True & is_satisfied==True 的数量
      - success_rate: success_count / total_pairs
      - origF_fixT_count: original_prediction==False & fixed_prediction==True 的数量
      - is_satisfied_count: is_satisfied==True 的数量
    """

    df = all_pairs.copy()

    # 统一布尔列
    def _to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y", "t"]
        )

    for col in ["original_prediction", "fixed_prediction", "is_satisfied"]:
        df[col] = _to_bool(df[col])

    # 各条件
    df["success"] = (~df["original_prediction"]) & df["fixed_prediction"] & df["is_satisfied"]
    df["origF_fixT"] = (~df["original_prediction"]) & df["fixed_prediction"]

    grouped = (
        df.groupby(["model", "N"], dropna=False)
          .agg(
              total_pairs=("pair_base_name", "size"),
              success_count=("success", "sum"),
              origF_fixT_count=("origF_fixT", "sum"),
              is_satisfied_count=("is_satisfied", "sum")
          )
          .reset_index()
    )
    grouped["success_rate"] = grouped["success_count"] / grouped["total_pairs"]

    # 保存
    if output_path:
        grouped.to_excel(output_path, index=False)
        print(f"[compute] Saved stats to {output_path}")

    return grouped


# 用法示例




if __name__ == '__main__':
    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL/prediction_result'
    # 原先英文版（若已存在可保留）
    all_pairs, out_dir = build_and_save_pair_predictions_from_instances(root_dir)

    # 中文列名版本
    # build_and_save_pair_predictions_from_instances_cn(root_dir)
    # all_pairs = pd.read_excel("pair_predictions_all_models.xlsx")
    result = compute_success_rate_with_is_satisfied(
        all_pairs,
        output_path=os.path.join(out_dir, "2sat_pairs_three_ways_success_rate_with_is_satisfied.xlsx")
    )
    print(result.head())