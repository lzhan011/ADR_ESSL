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


def build_and_save_pair_predictions_from_instances(root_dir: str, three_level_similar):
    """
    Scan all *_instances_res_cross_n.xlsx under:
      - root_dir/analysis
      - three_level_similar/analysis
    Build per-model pair results (original vs fixed) and a combined file.
    Output column names are in English.

    Pairing rules:
      - For files under root_dir: keep EXISTING pairing logic:
          key = remove "_RC2_fixed" then strip extension (full basename match)
      - For files under three_level_similar:
          filenames look like:
            cnf_k3_N10_L35_alpha3.5_inst966_medium_1_sim0.897.cnf
            cnf_k3_N10_L35_alpha3.5_inst966_RC2_fixed_medium_1_sim0.882.cnf
          Pair if (same inst number) AND (same level in {high, medium, low}),
          ignoring the sim value and the presence of "RC2_fixed_".

    Additionally, join is_satisfied from Assigned_value_satisfied_result_per_file.csv
    by (model/model_name, N, fixed_file/file_name).
    """
    import os, glob, re
    from pathlib import Path
    import pandas as pd

    def _strip_ext(s: str) -> str:
        return re.sub(r"\.[^.]+$", "", str(s))

    def _auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make sure the necessary columns exist with standard names:
        required: N, file_name, prediction_IS_SAT
        Try to normalize common variants if needed.
        """
        colmap = {}
        cols_lower = {c.lower(): c for c in df.columns}

        # N
        if "n" in cols_lower:
            colmap[cols_lower["n"]] = "N"
        elif "N" not in df.columns:
            raise KeyError("Missing column 'N'")

        # file_name
        if "file_name" in df.columns:
            pass
        elif "filename" in cols_lower:
            colmap[cols_lower["filename"]] = "file_name"
        elif "file" in cols_lower:
            colmap[cols_lower["file"]] = "file_name"
        else:
            # some tables use 'name' or 'input_file'
            if "name" in cols_lower:
                colmap[cols_lower["name"]] = "file_name"
            elif "input_file" in cols_lower:
                colmap[cols_lower["input_file"]] = "file_name"
            else:
                raise KeyError("Missing column 'file_name' (or alias)")

        # prediction_IS_SAT
        # normalize from variants like 'prediction', 'is_sat', 'pred_is_sat', etc.
        candidates = [
            "prediction_is_sat", "prediction", "is_sat", "pred_is_sat", "y_pred_is_sat"
        ]
        target_src = None
        for k in candidates:
            if k in cols_lower:
                target_src = cols_lower[k]
                break
        if target_src is None and "prediction_IS_SAT" not in df.columns:
            raise KeyError("Missing column 'prediction_IS_SAT' (or alias)")
        if target_src is not None:
            colmap[target_src] = "prediction_IS_SAT"

        if colmap:
            df = df.rename(columns=colmap)
        # final sanity
        need = ["N", "file_name", "prediction_IS_SAT"]
        for c in need:
            if c not in df.columns:
                raise KeyError(f"Missing expected column '{c}' after normalization")
        return df

    analysis_dir = os.path.join(root_dir, 'analysis')
    three_level_similar_analysis_dir = os.path.join(three_level_similar, 'analysis')
    out_dir = os.path.join(three_level_similar_analysis_dir, 'three_ways_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    root_excel_paths = glob.glob(os.path.join(analysis_dir, "*_instances_res_cross_n.xlsx"))
    three_level_excel_paths = glob.glob(os.path.join(three_level_similar_analysis_dir, "*_instances_res_cross_n.xlsx"))
    # 结合两个来源，一并处理（保留你原本注释掉的合并做法）
    excel_paths = root_excel_paths + three_level_excel_paths
    if not excel_paths:
        print(
            f"[pair-builder] No *_instances_res_cross_n.xlsx found in {analysis_dir} or {three_level_similar_analysis_dir}, skip.")
        return

    def is_three_level_book(xls_path: str) -> bool:
        # 属于 three_level_similar 这侧的表，就用新的配对键逻辑
        # 注意不同操作系统的大小写/分隔符
        return Path(xls_path).as_posix().startswith(Path(three_level_similar_analysis_dir).as_posix())

    # 为 three_level_similar 的文件名构造“配对键”：inst编号 + level
    # 例子：
    #   原件:  cnf_k3_N10_L35_alpha3.5_inst966_medium_1_sim0.897.cnf
    #   固定:  cnf_k3_N10_L35_alpha3.5_inst966_RC2_fixed_medium_1_sim0.882.cnf
    # 键形如： inst966|medium
    three_level_pat = re.compile(
        r"""_inst(?P<inst>\d+)_
            (?:(?:RC2_fixed_)?)
            (?P<level>high|medium|low)
            _""",
        re.IGNORECASE | re.VERBOSE
    )

    def make_three_level_pair_key(fname: str) -> str:
        s = str(fname)
        m = three_level_pat.search(s)
        if not m:
            # 回退策略：尽最大努力提取 inst 与 level；若失败，返回去扩展名去掉 RC2_fixed 后的简化名（能配上算运气）
            s2 = _strip_ext(s.replace("_RC2_fixed_", "_"))
            return s2
        inst = m.group("inst")
        level = m.group("level").lower()
        return f"inst{inst}|{level}"

    def make_root_pair_key(fname: str) -> str:
        # 原有逻辑：去掉 _RC2_fixed 后的去扩展名作为配对键
        return _strip_ext(str(fname).replace("_RC2_fixed", ""))

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

        # 标注 fixed
        df["__is_fixed__"] = df["file_name"].astype(str).str.contains("RC2_fixed", case=False, regex=False)

        # 根据来源构造“配对键”
        if is_three_level_book(xls_path):
            df["__pair_key__"] = df["file_name"].astype(str).apply(make_three_level_pair_key)
        else:
            df["__pair_key__"] = df["file_name"].astype(str).apply(make_root_pair_key)

        # 最小化列
        df_min = df[["N", "file_name", "prediction_IS_SAT", "__is_fixed__", "__pair_key__"]].copy()

        # Separate original vs fixed
        df_orig = df_min[~df_min["__is_fixed__"]].rename(columns={
            "file_name": "original_file",
            "prediction_IS_SAT": "original_prediction"
        })[["N", "__pair_key__", "original_file", "original_prediction"]]

        df_fix = df_min[df_min["__is_fixed__"]].rename(columns={
            "file_name": "fixed_file",
            "prediction_IS_SAT": "fixed_prediction"
        })[["N", "__pair_key__", "fixed_file", "fixed_prediction"]]

        # Join pairs on N + pair_key（注意：three_level 下 pair_key 已忽略 sim 差异，只关心 inst + level）
        pairs = pd.merge(df_orig, df_fix, on=["N", "__pair_key__"], how="inner")

        # Add model and base name（把 pair_key 作为 pair_base_name 输出，便于回查）
        pairs.insert(0, "model", model_name)
        pairs.insert(1, "pair_base_name", pairs["__pair_key__"])
        pairs.drop(columns=["__pair_key__"], inplace=True)

        # Save per-model file
        per_model_out = os.path.join(out_dir, f"{model_name}_pair_predictions.xlsx")
        pairs.to_excel(per_model_out, index=False)
        print(f"[pair-builder] Saved: {per_model_out} ({len(pairs)} pairs)")

        all_models_rows.append(pairs)

    if all_models_rows:
        all_pairs = pd.concat(all_models_rows, ignore_index=True)

        # ==== merge is_satisfied from per-file CSV (root_dir/analysis/three_ways_evaluation/Assigned_value_satisfied_result_per_file.csv) ====
        sat_csv = os.path.join(
            analysis_dir, "three_ways_evaluation", "Assigned_value_satisfied_result_per_file.csv"
        )
        sat_csv_three_similar_level = os.path.join(three_level_similar_analysis_dir, 'three_ways_evaluation', 'Assigned_value_satisfied_result_per_file.csv')
        if os.path.exists(sat_csv):
            try:
                df_sat = pd.read_csv(sat_csv)
                needed = ['model_name', 'N', 'file_name', 'is_satisfied']
                missing = [c for c in needed if c not in df_sat.columns]

                df_sat_three_similar_level  = pd.read_csv(sat_csv_three_similar_level)
                missing_three_similar_level = [c for c in needed if c not in df_sat_three_similar_level.columns]
                df_sat = pd.concat([df_sat, df_sat_three_similar_level], ignore_index=True)
                missing = missing + missing_three_similar_level
                if missing:
                    print(
                        f"[pair-builder] WARNING: columns {missing} not found in {sat_csv}; skip merging is_satisfied.")
                else:
                    df_sat = df_sat[needed].copy()
                    df_sat['model_name'] = df_sat['model_name'].astype(str)
                    df_sat['N'] = pd.to_numeric(df_sat['N'], errors='coerce')
                    df_sat = df_sat.dropna(subset=['N'])
                    df_sat['N'] = df_sat['N'].astype(int)

                    # Ensure boolean
                    if df_sat['is_satisfied'].dtype != bool:
                        df_sat['is_satisfied'] = (
                            df_sat['is_satisfied'].astype(str).str.strip().str.lower()
                            .isin(['true', '1', 'True', '1.0','TRUE', 'yes', 'y', 't'])
                        )

                    df_sat = df_sat.rename(columns={'model_name': 'model', 'file_name': 'fixed_file'})

                    all_pairs = pd.merge(
                        all_pairs, df_sat[['model', 'N', 'fixed_file', 'is_satisfied']],
                        on=['model', 'N', 'fixed_file'], how='left'
                    )
                    print(f"[pair-builder] Merged is_satisfied from {sat_csv}")
            except Exception as e:
                print(f"[pair-builder] Failed to merge is_satisfied from {sat_csv}: {e}")
        else:
            print(f"[pair-builder] WARNING: per-file CSV not found: {sat_csv}; skip merging is_satisfied.")

        # Save combined file
        all_out = os.path.join(out_dir, "pair_predictions_all_models.xlsx")
        all_pairs.to_excel(all_out, index=False)
        print(f"[pair-builder] Saved combined results: {all_out} ({len(all_pairs)} pairs)")

        return all_pairs, out_dir
    else:
        print("[pair-builder] No pair results to combine.")
        return None, out_dir


import pandas as pd


def compute_3_ways_evaluation(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    对 all_pairs 计算每个 (model, N) 的统计：
      - success_count: original_prediction==False & fixed_prediction==True & is_satisfied==True 的数量
      - 3_ways_evaluation: success_count / 总数
      - origF_fixT_count: original_prediction==False & fixed_prediction==True 的数量
      - Assignment_satisfied_count: is_satisfied==True 的数量
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
        Assignment_satisfied_count = mask_satisfied.sum()

        prob = success_count / total if total > 0 else 0.0

        results.append({
            'model': m,
            'N': n,
            'total_pairs': total,
            'success_count': success_count,
            'origF_fixT_count': origF_fixT_count,
            'Assignment_satisfied_count': Assignment_satisfied_count,
            'Assignment_satisfied_rate': Assignment_satisfied_count/total,
            '3_ways_evaluation': prob
        })
    return pd.DataFrame(results).sort_values(['model', 'N'])



def random_sample(all_pairs: pd.DataFrame,
                  out_dir: str,
                  sample_sizes = [70, 140, 210, 280],
                  random_state: int = 42) -> None:
    """
    对每个 N：
      1) 汇总该 N 下所有 model 的 pairs，做全量 union 并去重；
      2) 在此 union 上做随机采样（70/140/210/280，若不足则取最小值）；
      3) 用该采样集合，分别对每个 model 在该 N 下的子集计算指标；
    最终每个样本量写入一个 sheet；另附 all_pairs 的全量统计；
    并新增一个拼接 sheet（samples_wide），把所有样本量横向合并（同一 model、N 对齐）。
    """
    import os
    import pandas as pd
    import numpy as np
    from functools import reduce

    # 简单校验
    if all_pairs is None or not isinstance(all_pairs, pd.DataFrame) or len(all_pairs) == 0:
        print("[pair-builder] SKIP sampling: all_pairs is empty or None.")
        return

    # 必要列检查 & 构造跨模型可对齐的“pair 唯一键”
    needed_any = {'model', 'N', 'original_file', 'fixed_file'}
    alt_key_ok = 'pair_base_name' in all_pairs.columns
    if not needed_any.issubset(all_pairs.columns) and not alt_key_ok:
        raise KeyError(
            "需要列 {model, N, original_file, fixed_file}（或至少提供 pair_base_name）以便跨模型对齐采样。"
            f" 当前列: {list(all_pairs.columns)}"
        )

    df = all_pairs.copy()

    # 构造 pair 唯一键（优先用 original_file + fixed_file，更稳健；否则退回 pair_base_name）
    if {'original_file', 'fixed_file'}.issubset(df.columns):
        df['__pair_uid__'] = df['original_file'].astype(str) + '||' + df['fixed_file'].astype(str)
    else:
        df['__pair_uid__'] = df['pair_base_name'].astype(str)

    # 预先把全量表算好（不依赖采样）
    df_prob_all = compute_3_ways_evaluation(df)

    samples_out = os.path.join(out_dir, "three_ways_evaluation_3_ways_evaluation_by_model_N_samples.xlsx")

    # 收集每个样本量整理后的 DataFrame，用于后续横向拼接
    per_size_tables: dict[int, pd.DataFrame] = {}

    # 写多 sheet
    with pd.ExcelWriter(samples_out, engine="xlsxwriter") as writer:
        for size in sample_sizes:
            rows = []

            # 遍历每个 N
            for N_val in sorted(df['N'].unique()):
                dfN = df[df['N'] == N_val]

                # 该 N 下的“全模型 union 去重”
                dfN_unique_pairs = dfN.drop_duplicates(subset=['__pair_uid__'])
                unique_count = len(dfN_unique_pairs)
                k = min(size, unique_count)

                # 为了让每个 (样本量, N) 的采样可复现，用 (random_state, size, N) 组合成 seed
                seed_str = f"{random_state}-{size}-{int(N_val)}"
                rs_seed = (abs(np.int64(np.frombuffer(seed_str.encode('utf-8'), dtype=np.uint8).sum())) % (2**32 - 1)) + 1

                if k > 0:
                    sampled_uids = dfN_unique_pairs['__pair_uid__'].sample(n=k, random_state=int(rs_seed))
                else:
                    sampled_uids = pd.Series([], dtype=dfN_unique_pairs['__pair_uid__'].dtype)

                # 在该 N 下，针对每个 model，用相同的 sampled_uids 过滤后计算
                for m in sorted(dfN['model'].unique()):
                    dfN_m = dfN[dfN['model'] == m]
                    dfN_m_sample = dfN_m[dfN_m['__pair_uid__'].isin(sampled_uids)]

                    if len(dfN_m_sample) == 0:
                        rows.append({
                            'model': m,
                            'N': int(N_val),
                            'total_pairs': 0,
                            'success_count': 0,
                            'origF_fixT_count': 0,
                            'Assignment_satisfied_count': 0,
                            '3_ways_evaluation': 0.0
                        })
                    else:
                        df_one = compute_3_ways_evaluation(dfN_m_sample)
                        if len(df_one) == 1:
                            row = df_one.iloc[0].to_dict()
                            row['total_pairs'] = int(len(dfN_m_sample))
                            row['N'] = int(N_val)
                            rows.append(row)
                        else:
                            for _, r in df_one.iterrows():
                                rr = r.to_dict()
                                rr['total_pairs'] = int(len(dfN_m_sample))
                                rr['N'] = int(N_val)
                                rows.append(rr)

            # ===== 汇总为 DataFrame，并按 (model, N) 排序（同 model 内按 N 升序） =====
            df_size = pd.DataFrame(rows)
            if not df_size.empty:
                df_size['N'] = pd.to_numeric(df_size['N'], errors='coerce').astype('Int64')
                df_size = df_size[['model', 'N', 'total_pairs',
                                   'success_count', 'origF_fixT_count',
                                   'Assignment_satisfied_count', '3_ways_evaluation']]

                # 新增：Assignment_satisfied_rate = Assignment_satisfied_count / total_pairs
                # 避免除零与 NaN
                denom = df_size['total_pairs'].replace({0: np.nan})
                df_size['Assignment_satisfied_rate'] = (df_size['Assignment_satisfied_count'] / denom).fillna(0.0)

                # 自定义模型排序（可改顺序；未包含的模型会排在最后并保持 NaN 类别顺序）
                model_order = ['deepseek-reasoner','gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4.1',
                               'gpt-4o', 'gpt-5', 'o1', 'o3-mini', 'gpt-3.5-turbo-0125', 'chatgpt-4o-latest']
                cat = pd.CategoricalDtype(categories=model_order, ordered=True)
                df_size['model'] = df_size['model'].astype('string').astype(cat)

                df_size = df_size.sort_values(by=['model', 'N'], ascending=[True, True])

            # 写入当前样本量的 sheet（把 rate 放在计数后、3_ways_evaluation 前）
            df_size = df_size[['model', 'N',
                               'success_count', 'origF_fixT_count',
                               'Assignment_satisfied_count', 'Assignment_satisfied_rate',
                               '3_ways_evaluation', 'total_pairs']]
            df_size.to_excel(writer, sheet_name=f"n_{size}", index=False)

            # ====== 为“拼接”准备：复制一份并重命名为带后缀的列 ======
            if not df_size.empty:
                df_wide = df_size.copy()
                rename_map = {
                    'total_pairs': f'total_pairs_n{size}',
                    'success_count': f'success_count_n{size}',
                    'origF_fixT_count': f'origF_fixT_count_n{size}',
                    'Assignment_satisfied_count': f'Assignment_satisfied_count_n{size}',
                    'Assignment_satisfied_rate': f'Assignment_satisfied_rate_n{size}',
                    '3_ways_evaluation': f'3_ways_evaluation_n{size}',
                }
                df_wide = df_wide.rename(columns=rename_map)
                per_size_tables[size] = df_wide
            else:
                per_size_tables[size] = pd.DataFrame(
                    columns=['model', 'N',
                             f'success_count_n{size}', f'origF_fixT_count_n{size}',
                             f'Assignment_satisfied_count_n{size}', f'Assignment_satisfied_rate_n{size}',
                             f'3_ways_evaluation_n{size}', f'total_pairs_n{size}']
                )

        # ===== 合并所有样本量为一个“宽表”并写入新 sheet =====
        if per_size_tables:
            sizes_in_order = [s for s in sample_sizes if s in per_size_tables]
            df_wide_all = reduce(
                lambda left, right: pd.merge(left, right, on=['model', 'N'], how='outer'),
                [per_size_tables[s] for s in sizes_in_order]
            )

            # 排序
            model_order = ['deepseek-reasoner','gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4.1',
                           'gpt-4o', 'gpt-5', 'o1', 'o3-mini',
                           'gpt-3.5-turbo-0125', 'chatgpt-4o-latest']
            cat = pd.CategoricalDtype(categories=model_order, ordered=True)
            df_wide_all['model'] = df_wide_all['model'].astype('string').astype(cat)
            df_wide_all['N'] = pd.to_numeric(df_wide_all['N'], errors='coerce').astype('Int64')
            df_wide_all = df_wide_all.sort_values(by=['model', 'N'], ascending=[True, True])

            # 列顺序：model, N, success_* → origF_fixT_* → Assignment_satisfied_count_* → Assignment_satisfied_rate_* → 3_ways_* → total_pairs_*
            ordered_cols = ['model', 'N']
            for metric in ['success_count', 'origF_fixT_count', 'Assignment_satisfied_count',
                           'Assignment_satisfied_rate', '3_ways_evaluation', 'total_pairs']:
                for s in sizes_in_order:
                    ordered_cols.append(f"{metric}_n{s}")

            ordered_cols = [c for c in ordered_cols if c in df_wide_all.columns]
            df_wide_all = df_wide_all[ordered_cols]

            df_wide_all.to_excel(writer, sheet_name="samples_wide", index=False)

        # 附加全量统计
        df_prob_all.to_excel(writer, sheet_name="all_pairs", index=False)

    print(f"[pair-builder] Saved sampled stats (multi-sheet): {samples_out}")





if __name__ == '__main__':
    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    three_level_similar = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result'
    # —— 你可以在主流程最后追加这一行来生成结果（不会影响你现有逻辑）——
    all_pairs, out_dir = build_and_save_pair_predictions_from_instances(root_dir, three_level_similar)

    df_prob = compute_3_ways_evaluation(all_pairs)
    print(df_prob)
    # 保存结果
    df_prob.to_excel(os.path.join(out_dir, "three_ways_evaluation_3_ways_evaluation_by_model_N.xlsx"), index=False)

    random_sample(all_pairs, out_dir)