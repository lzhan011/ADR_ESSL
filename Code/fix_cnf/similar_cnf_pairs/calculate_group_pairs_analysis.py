import json
import os
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Iterable

# =========================
# 常量
# =========================
SIM_LEVELS = ['high', 'medium', 'low']
PAIR_LEVELS_ORDER = ['Original', 'high', 'medium', 'low']

# 预置列，确保即便空表也不报 KeyError
GROUP_COLS = [
    'model', 'N', 'group_base_name',
    'available_pair_count', 'correct_pair_count', 'group_score', 'available_pairs',
    # 新增：方便汇总
    'group_all_correct', 'pair_total', 'pair_correct'
]
DETAIL_COLS = [
    'model', 'N', 'group_base_name',
    'pair_level', 'pair_exists', 'pair_fully_correct', 'file_name'
]
SUMMARY_BY_N_COLS = [
    'model', 'N',
    'total_groups', 'valid_groups', 'weight_group_score',
    # 新增的两类指标
    'all_correct_groups', 'all_groups_valid', 'group_level_ADR',
    'total_pairs', 'correct_pairs', 'pair_level_ADR'
]


# =========================
# 基础 IO
# =========================
def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f) or {}

def decide_Separated_Correct(pair_prediction: Dict[str, Any]) -> bool:
    """
    你原有的“pair都正确”的判定：after_fix=True 且 before_fix=False
    """
    try:
        return bool(pair_prediction.get('Predictions_after_fix')) and (pair_prediction.get('Predictions_before_fix') is False)
    except Exception:
        return False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_excel_with_sheets(output_file: str,
                            dataframes: List[Tuple[str, pd.DataFrame]]) -> None:
    ensure_dir(os.path.dirname(output_file))
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in dataframes:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# =========================
# 名称与集合工具
# =========================
def get_base_name_from_sim_file(file_name: str, sim_level: str) -> str:
    """
    从 'xxx_high_...' / 'xxx_medium_...' / 'xxx_low_...' 提取 base_name 与 original 对齐。
    若未匹配到 sim_level，返回原 file_name。
    """
    try:
        idx = file_name.index(sim_level)
        return file_name[:max(idx - 1, 0)]
    except ValueError:
        return file_name

def union_keys(dicts: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for d in dicts:
        keys |= set(d.keys())
    return sorted(keys)


# =========================
# 读取并标准化数据
# =========================
def load_original_predictions(original_dir: str, model: str) -> Dict[str, Dict[str, Any]]:
    """
    返回: { N: { file_name: pair_dict, ... }, ... }
    """
    path = os.path.join(original_dir, f"{model}_original_pairs_prediction_res.json")
    return load_json(path)

def read_similar_predictions(three_level_dir: str, model: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    返回:
    {
      'high':   { N: { base_name: pair_dict, ... }, ... },
      'medium': { N: { base_name: pair_dict, ... }, ... },
      'low':    { N: { base_name: pair_dict, ... }, ... }
    }
    """
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sim in SIM_LEVELS:
        raw = load_json(os.path.join(three_level_dir, f"{model}_{sim}_pairs_prediction_res.json"))
        norm_by_n: Dict[str, Dict[str, Any]] = {}
        for N, one_n_map in (raw or {}).items():
            base_map: Dict[str, Any] = {}
            for fn, v in (one_n_map or {}).items():
                v = v or {}
                base = get_base_name_from_sim_file(fn, sim)
                if v != {}:
                    v = dict(v)
                    v['file_name'] = fn
                    v['sim_level'] = sim
                base_map[base] = v
            norm_by_n[N] = base_map
        result[sim] = norm_by_n
    return result

def collect_all_Ns(original_data: Dict[str, Any],
                   similar_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    取 original 与 three-level 的 N 的并集，确保“所有 N 都覆盖”。
    """
    ns = set(original_data.keys())
    for sim in SIM_LEVELS:
        ns |= set(similar_data.get(sim, {}).keys())
    return sorted(ns)

def collect_all_base_names_for_N(original_map: Dict[str, Any],
                                 high_map: Dict[str, Any],
                                 med_map: Dict[str, Any],
                                 low_map: Dict[str, Any]) -> List[str]:
    return union_keys([original_map, high_map, med_map, low_map])

def build_groups_for_N(N: str,
                       original_data: Dict[str, Dict[str, Any]],
                       three_level_data: Dict[str, Dict[str, Dict[str, Any]]]
                       ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    返回:
    {
      base_name: {
        'Original': {...} or {},
        'high': {...} or {},
        'medium': {...} or {},
        'low': {...} or {}
      }, ...
    }
    """
    original_map = (original_data.get(N, {}) or {})
    high_map = (three_level_data.get('high', {}).get(N, {}) or {})
    med_map = (three_level_data.get('medium', {}).get(N, {}) or {})
    low_map = (three_level_data.get('low', {}).get(N, {}) or {})
    bases = collect_all_base_names_for_N(original_map, high_map, med_map, low_map)

    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for base in bases:
        groups[base] = {
            'Original': original_map.get(base, {}),
            'high': high_map.get(base, {}),
            'medium': med_map.get(base, {}),
            'low': low_map.get(base, {}),
        }
    return groups


# =========================
# 规则：pair 是否“都正确”
# =========================
def is_pair_fully_correct(pair_prediction: Dict[str, Any]) -> bool:
    """
    这里按你的定义：必须存在 Predictions_after_fix / Predictions_before_fix，并调用 decide_Separated_Correct
    """
    if not pair_prediction:
        return False
    if "Predictions_after_fix" in pair_prediction and "Predictions_before_fix" in pair_prediction:
        return decide_Separated_Correct(pair_prediction)
    else:
        return False


# =========================
# 计分
# =========================
def score_one_group(model: str,
                    N: str,
                    base_name: str,
                    parts: Dict[str, Dict[str, Any]]
                    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    available_levels: List[str] = []
    avail_cnt = 0
    correct_cnt = 0

    for level in PAIR_LEVELS_ORDER:
        pair_dict = parts.get(level, {}) or {}
        exists = (pair_dict != {})
        fully_correct = False
        if exists:
            fully_correct = is_pair_fully_correct(pair_dict)
            avail_cnt += 1
            if fully_correct:
                correct_cnt += 1
            available_levels.append(level)

        pair_rows.append({
            'model': model,
            'N': N,
            'group_base_name': base_name,
            'pair_level': level,
            'pair_exists': exists,
            'pair_fully_correct': fully_correct,
            'file_name': pair_dict.get('file_name') if exists else None
        })

    group_score: Optional[float] = (correct_cnt / avail_cnt) if avail_cnt > 0 else None
    group_all_correct = int(avail_cnt > 0 and correct_cnt == avail_cnt)

    group_row = {
        'model': model,
        'N': N,
        'group_base_name': base_name,
        'available_pair_count': avail_cnt,
        'correct_pair_count': correct_cnt,
        'group_score': group_score,
        'available_pairs': ','.join(available_levels) if available_levels else '',
        # 新增便于汇总
        'group_all_correct': group_all_correct,
        'pair_total': avail_cnt,
        'pair_correct': correct_cnt
    }
    return group_row, pair_rows

def empty_df_with_cols(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in cols})

def score_all_groups_for_N(model: str,
                           N: str,
                           groups_for_n: Dict[str, Dict[str, Dict[str, Any]]]
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回: (group_df, detail_df, summary_df_by_N)
    均保证即便没有任何行也具备预置列，避免 KeyError。
    """
    if not groups_for_n:
        group_df = empty_df_with_cols(GROUP_COLS)
        detail_df = empty_df_with_cols(DETAIL_COLS)
        summary_df = pd.DataFrame([{
            'model': model, 'N': N,
            'total_groups': 0, 'valid_groups': 0, 'weight_group_score': None,
            'all_correct_groups': 0, 'all_groups_valid': 0, 'group_level_ADR': None,
            'total_pairs': 0, 'correct_pairs': 0, 'pair_level_ADR': None
        }], columns=SUMMARY_BY_N_COLS)
        return group_df, detail_df, summary_df

    group_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for base, parts in groups_for_n.items():
        g_row, p_rows = score_one_group(model, N, base, parts)
        group_rows.append(g_row)
        detail_rows.extend(p_rows)

    group_df = pd.DataFrame(group_rows, columns=GROUP_COLS)
    detail_df = pd.DataFrame(detail_rows, columns=DETAIL_COLS)

    # 仅统计 available_pair_count > 0 的 group 作为“有效组”
    if not group_df.empty:
        valid_df = group_df[group_df['available_pair_count'] > 0].copy()
        avg_score = float(valid_df['group_score'].mean()) if not valid_df.empty else None
        total_groups = len(group_df)
        valid_groups = int(valid_df.shape[0])

        # 新增：组“全对率”
        all_correct_groups = int(valid_df['group_all_correct'].sum())
        all_groups_valid = valid_groups
        group_level_ADR = (all_correct_groups / all_groups_valid) if all_groups_valid > 0 else None

        # 新增：pair 层面的整体准确率
        total_pairs = int(valid_df['pair_total'].sum())
        correct_pairs = int(valid_df['pair_correct'].sum())
        pair_level_ADR = (correct_pairs / total_pairs) if total_pairs > 0 else None
    else:
        avg_score, total_groups, valid_groups = None, 0, 0
        all_correct_groups, all_groups_valid, group_level_ADR = 0, 0, None
        total_pairs, correct_pairs, pair_level_ADR = 0, 0, None

    summary_df = pd.DataFrame([{
        'model': model, 'N': N,
        'total_groups': total_groups,
        'valid_groups': valid_groups,
        'weight_group_score': avg_score,
        'all_correct_groups': all_correct_groups,
        'all_groups_valid': all_groups_valid,
        'group_level_ADR': group_level_ADR,
        'total_pairs': total_pairs,
        'correct_pairs': correct_pairs,
        'pair_level_ADR': pair_level_ADR
    }], columns=SUMMARY_BY_N_COLS)

    return group_df, detail_df, summary_df


# =========================
# 单模型流程
# =========================
def analyze_one_model_for_N(original_dir: str,
                            three_level_dir: str,
                            model: str,
                            N: str,
                            per_model_output_dir: Optional[str] = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    返回 (group_df, detail_df, summary_df_by_N, out_path)
    若 per_model_output_dir 为 None，则不写单独 Excel。
    """
    original_data = load_original_predictions(original_dir, model)
    similar_data = read_similar_predictions(three_level_dir, model)
    groups_for_n = build_groups_for_N(N, original_data, similar_data)

    group_df, detail_df, summary_df = score_all_groups_for_N(model, N, groups_for_n)

    out_path = None
    if per_model_output_dir:
        ensure_dir(per_model_output_dir)
        out_path = os.path.join(per_model_output_dir, f"group_analysis_{model}_{N}_pairs_prediction_res.xlsx")
        write_excel_with_sheets(out_path, [
            ('group_scores', group_df),
            ('pair_details', detail_df),
            ('summary', summary_df),
        ])
    return group_df, detail_df, summary_df, out_path

def analyze_one_model_all_N(original_dir: str,
                            three_level_dir: str,
                            model: str,
                            per_model_output_dir: Optional[str] = None,
                            N_list: Optional[List[str]] = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    聚合该模型（全部或指定 N）的结果。
    返回 (group_all, detail_all, summary_by_N_all, paths_written)
    """
    original_data = load_original_predictions(original_dir, model)
    similar_data = read_similar_predictions(three_level_dir, model)

    if N_list is None:
        N_list = collect_all_Ns(original_data, similar_data)

    all_groups, all_details, all_summaries = [], [], []
    paths = []

    for N in N_list:
        gdf, ddf, sdf, path = analyze_one_model_for_N(
            original_dir, three_level_dir, model, N, per_model_output_dir
        )
        if not gdf.empty:
            all_groups.append(gdf)
        if not ddf.empty:
            all_details.append(ddf)
        if not sdf.empty:
            all_summaries.append(sdf)
        if path:
            paths.append(path)

    group_all = pd.concat(all_groups, ignore_index=True) if all_groups else empty_df_with_cols(GROUP_COLS)
    detail_all = pd.concat(all_details, ignore_index=True) if all_details else empty_df_with_cols(DETAIL_COLS)
    summary_by_N_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else empty_df_with_cols(SUMMARY_BY_N_COLS)

    return group_all, detail_all, summary_by_N_all, paths


# =========================
# 跨模型聚合与总输出
# =========================
def summarize_by_model_and_N(group_all_models: pd.DataFrame) -> pd.DataFrame:
    """
    计算“每个模型 × 每个 N”的总组数、有效组数（available_pair_count>0）与平均 group_score。
    这里保留旧的三列；新增的两类指标已经在 summary_by_model_N 中给出。
    """
    cols = ['model', 'N', 'total_groups', 'valid_groups', 'weight_group_score']
    if group_all_models.empty:
        return pd.DataFrame(columns=cols)

    df = group_all_models.copy()

    total = (df.groupby(['model', 'N'])
               .size()
               .reset_index(name='total_groups'))

    valid = df[df['available_pair_count'] > 0]
    if valid.empty:
        out = total.copy()
        out['valid_groups'] = 0
        out['weight_group_score'] = None
        return out[cols]

    agg = (valid.groupby(['model', 'N'])
                 .agg(valid_groups=('group_score', 'size'),
                      weight_group_score=('group_score', 'mean'))
                 .reset_index())

    out = pd.merge(total, agg, on=['model', 'N'], how='left')
    out['valid_groups'] = out['valid_groups'].fillna(0).astype(int)
    out['weight_group_score'] = out['weight_group_score'].astype(float)

    return out[cols]

def summarize_overall(group_all_models: pd.DataFrame) -> pd.DataFrame:
    """
    计算跨模型整体的平均 group_score（仅对 available_pair_count>0 的组）。
    """
    if group_all_models.empty:
        return pd.DataFrame([{
            'overall_total_groups': 0,
            'overall_valid_groups': 0,
            'overall_weight_group_score': None
        }])
    df = group_all_models
    total = len(df)
    valid = df[df['available_pair_count'] > 0]
    return pd.DataFrame([{
        'overall_total_groups': total,
        'overall_valid_groups': int(valid.shape[0]),
        'overall_weight_group_score': float(valid['group_score'].mean()) if not valid.empty else None
    }])

def analyze_models(original_dir: str,
                   three_level_dir: str,
                   models: List[str],
                   output_root: str,
                   N_list: Optional[List[str]] = None,
                   write_per_model_files: bool = True,
                   all_in_one_filename: str = "ALL_models_ALL_N_group_pairs_analysis.xlsx"
                   ) -> str:
    """
    跑“所有模型 × N（可选）”，并写 1 个大 Excel 汇总。
    返回：大 Excel 的输出路径。
    """
    ensure_dir(output_root)

    group_all_models_list, detail_all_models_list, summary_by_N_all_models_list = [], [], []

    for m in models:
        per_model_dir = os.path.join(output_root, m) if write_per_model_files else None
        g_all, d_all, s_all, _ = analyze_one_model_all_N(
            original_dir, three_level_dir, m, per_model_dir, N_list
        )
        if not g_all.empty:
            group_all_models_list.append(g_all)
        if not d_all.empty:
            detail_all_models_list.append(d_all)
        if not s_all.empty:
            summary_by_N_all_models_list.append(s_all)

    # 合并所有模型
    all_group_scores = (pd.concat(group_all_models_list, ignore_index=True)
                        if group_all_models_list else empty_df_with_cols(GROUP_COLS))
    all_pair_details = (pd.concat(detail_all_models_list, ignore_index=True)
                        if detail_all_models_list else empty_df_with_cols(DETAIL_COLS))
    summary_by_model_N = (pd.concat(summary_by_N_all_models_list, ignore_index=True)
                          if summary_by_N_all_models_list else empty_df_with_cols(SUMMARY_BY_N_COLS))

    # 排序：先按 model 升序，再按 N 的数值升序
    if not summary_by_model_N.empty:
        summary_by_model_N['N_num'] = pd.to_numeric(summary_by_model_N['N'], errors='coerce')
        summary_by_model_N = summary_by_model_N.sort_values(by=['model', 'N_num']).drop(columns=['N_num'])

    # 衍生总结（按 model × N 的平均 group_score）
    summary_by_model_df = summarize_by_model_and_N(all_group_scores)
    overall_summary_df = summarize_overall(all_group_scores)

    # 写大 Excel
    all_out_path = os.path.join(output_root, all_in_one_filename)
    write_excel_with_sheets(all_out_path, [
        ('all_group_scores', all_group_scores),
        ('all_pair_details', all_pair_details),
        ('summary_by_model_N', summary_by_model_N),
        ('summary_by_model', summary_by_model_df),
        ('overall_summary', overall_summary_df),
    ])
    return all_out_path


# =========================
# 主入口
# =========================
if __name__ == '__main__':
    # === 路径设置（按需修改）===
    ORIGINAL_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N/analysis/pairs_prediction_res'
    THREE_LEVEL_DIR = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result/analysis/pairs_prediction_res'
    OUTPUT_ROOT = os.path.join(ORIGINAL_DIR, 'group_analysis')

    # === 模型与 N 选择（按需修改）===
    MODEL_LIST = ['gpt-5',  'deepseek-reasoner', 'gpt-3.5-turbo', 'o1', 'gpt-4.1', 'gpt-4o', 'gpt-4-turbo', 'o3-mini']

    N_LIST = None                # 跑“并集”上的全部 N
    # N_LIST = ['25']            # 只跑特定 N
    # === 执行 ===
    all_excel_path = analyze_models(
        original_dir=ORIGINAL_DIR,
        three_level_dir=THREE_LEVEL_DIR,
        models=MODEL_LIST,
        output_root=OUTPUT_ROOT,
        N_list=N_LIST,
        write_per_model_files=True,  # 同时输出每模型/每N的小文件
        all_in_one_filename="ALL_models_ALL_N_group_pairs_analysis.xlsx"
    )
    print(f"[OK] Wrote all-in-one Excel: {all_excel_path}")
