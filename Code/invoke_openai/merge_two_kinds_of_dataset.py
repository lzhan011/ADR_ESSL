import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from draw_legend import *

# ------------------------ 配置路径 ------------------------
ONE_DATASET = r'/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/draw_o1_phase_transition_figures/figures_comparison_1800_instances/draw_o1_cnf_alpha_3_6_N_75/per_file_predictions.xlsx'
SECOND_DATASET = r'/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/figures_comparison/cnf_results_openai_/per_file_predictions.xlsx'
OUT_DIR = os.path.join(os.path.dirname(ONE_DATASET), 'combined_metrics')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 固定横轴范围与步长 ----
X_MIN = 3.0
X_MAX = 5.5
X_STEP = 0.5
ALPHAS_FIXED = np.round(np.arange(X_MIN, X_MAX + 1e-9, X_STEP), 1)

# ------------------------ 论文友好样式 ------------------------
def apply_paper_style(
    base_font=26,
    line_width=3.0,
    marker_size=10.0,
    legend_cols=4
):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font-2,
        "ytick.labelsize": base_font-2,
        "legend.fontsize": base_font-2,
        "figure.titlesize": base_font+2,
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "savefig.dpi": 600,
    })
    return legend_cols

LEGEND_NCOLS = apply_paper_style(
    base_font=26,
    line_width=3.0,
    marker_size=10.0,
    legend_cols=4
)

# ------------------------ 工具函数 ------------------------
def get_model_for_match(model_original: str) -> str:
    model_list = [
        'gpt-3.5-turbo-0125',
        'deepseek-chat',
        'deepseek-reasoner',
        "claude-3-opus",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
        "claude-3-5-haiku",
        "o3-mini",
        "gpt-4o-latest",
        "gpt-4.1",
        "CDCL",                      # <-- 新增
    ]
    s = str(model_original)
    s_low = s.lower()

    if s.endswith('o1') or s.endswith('openai_prediction_o1'):
        return "o1"
    if s.endswith('gpt-5_no_batch'):
        return "gpt-5"

    for m in model_list:
        if m.lower() in s_low:
            return m
    return ""

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    low = s.astype(str).str.strip().str.lower()
    mapping = {'true': True, '1': True, 'false': False, '0': False, 'yes': True, 'no': False}
    return low.map(mapping)

def _safe_div(num, den):
    return num / den if den else np.nan

def _f1(p, r):
    if p is None or r is None:
        return np.nan
    if (p is np.nan) or (r is np.nan) or (p + r) == 0:
        return np.nan
    return 2 * p * r / (p + r)

def _mcc_from_counts(tp, fp, fn, tn):
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return np.nan
    return (tp * tn - fp * fn) / math.sqrt(denom)

def _disp_label(name: str) -> str:
    return name.lstrip('_')

# ------------------------ 读入与标准化 ------------------------
def load_per_file_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    needed = {'model', 'alpha', 'ground_truth_sat', 'predicted_sat', 'branches_number'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'文件缺少必要列: {missing} in {path}')

    df['ground_truth_sat'] = _to_bool_series(df['ground_truth_sat'])
    df['predicted_sat']    = _to_bool_series(df['predicted_sat'])
    df['alpha']            = pd.to_numeric(df['alpha'], errors='coerce').round(1)
    df['branches_number']  = pd.to_numeric(df['branches_number'], errors='coerce')

    if 'model_for_match' not in df.columns:
        df['model_for_match'] = df['model'].astype(str).apply(get_model_for_match)
    else:
        df['model_for_match'] = df['model_for_match'].astype(str).replace({'': np.nan}).fillna(
            df['model'].astype(str).apply(get_model_for_match)
        )

    df = df[df['model_for_match']!= ""]
    return df

def combine_datasets(paths) -> pd.DataFrame:
    frames = [load_per_file_xlsx(p) for p in paths]
    return pd.concat(frames, ignore_index=True)

# ------------------------ 聚合与指标计算 ------------------------
def aggregate_by_model_alpha(df: pd.DataFrame) -> pd.DataFrame:
    def _block(g: pd.DataFrame) -> pd.Series:
        n  = len(g)
        gt = g['ground_truth_sat'].to_numpy(dtype=bool)
        pr = g['predicted_sat'].to_numpy(dtype=bool)

        tp = int((gt & pr).sum())
        fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        tn = int((~gt & ~pr).sum())

        acc = _safe_div(tp + tn, n)

        # SAT 为正类
        p_sat = _safe_div(tp, tp + fp)
        r_sat = _safe_div(tp, tp + fn)
        f_sat = _f1(p_sat, r_sat)

        # UNSAT 为正类
        p_uns = _safe_div(tn, tn + fn)
        r_uns = _safe_div(tn, tn + fp)
        f_uns = _f1(p_uns, r_uns)

        # phase prob（预测为 SAT 的比例）
        sat_prob = _safe_div(tp + fp, n)

        # phase difficulty（branches_number 的中位数）
        med_br = np.nanmedian(g['branches_number'].to_numpy(dtype=float)) if g['branches_number'].notna().any() else np.nan

        # --- 新增：MCC ---
        mcc = _mcc_from_counts(tp, fp, fn, tn)

        return pd.Series({
            'n': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'accuracy': acc,
            'precision_sat': p_sat, 'recall_sat': r_sat, 'f1_sat': f_sat,
            'precision_unsat': p_uns, 'recall_unsat': r_uns, 'f1_unsat': f_uns,
            'sat_prob_pred': sat_prob,
            'median_branches_pred': med_br,
            'mcc': mcc,   # <--- 计算好的 MCC
        })

    out = (df.groupby(['model_for_match', 'alpha'], as_index=False)
             .apply(_block)
             .reset_index(drop=True))
    return out

# ------------------------ 画 3×3 九宫格（放大字体版） ------------------------
def plot_3x3(agg: pd.DataFrame, out_dir: str, ALPHAS_FIXED, file_stub: str = 'combined_metrics_3x3'):
    # 家族与样式映射
    CLAUDE_MODELS        = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus", "claude-sonnet-4"]
    DEEPSEEK_MODELS      = ["deepseek-chat", "deepseek-reasoner"]
    OPENAI_SPECIAL_RED   = ["gpt-5", "o1", "o3-mini"]
    OPENAI_OTHER_YELLOW  = ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]

    ORDER_LIST = CLAUDE_MODELS + DEEPSEEK_MODELS + OPENAI_SPECIAL_RED + OPENAI_OTHER_YELLOW + ["CDCL"]

    FAMILY_STYLE = {
        "claude"     : dict(color="#7b2cbf", linestyle="--"),
        "deepseek"   : dict(color="#2ca02c", linestyle=(0, (10, 6))),
        "openai_red" : dict(color="#d62728", linestyle="-"),
        "openai_yel" : dict(color="#ffbf00", linestyle="-"),
    }
    CLAUDE_TRIANGLES  = ["^", "v", "<", ">"]
    DEEPSEEK_MARKERS  = ["*", "s"]
    OPENAI_RED_MARKS  = ["o", "D", "X"]
    OPENAI_YEL_MARKS  = ["P", "h", "d"]

    def _which_openai_palette(name_lower: str) -> str:
        return "openai_red" if any(k in name_lower for k in ["gpt-5", "o1", "o3-mini"]) else "openai_yel"

    def get_style(model_name: str, fallback_idx: int = 0):
        name = (model_name or "").lower()
        if name == "cdcl":
            return {"color": "black", "linestyle": "-", "marker": "*"}

        if "claude" in name:
            fam, markers = "claude", CLAUDE_TRIANGLES
        elif "deepseek" in name:
            fam, markers = "deepseek", DEEPSEEK_MARKERS
        elif ("gpt" in name) or ("o" in name) or ("openai" in name):
            fam = _which_openai_palette(name)
            markers = OPENAI_RED_MARKS if fam == "openai_red" else OPENAI_YEL_MARKS
        else:
            fam, markers = "openai_yel", OPENAI_YEL_MARKS

        style = {**FAMILY_STYLE[fam]}
        style["marker"] = markers[fallback_idx % len(markers)]
        return style

    def order_models(models: list) -> list:
        idx = {m: i for i, m in enumerate(ORDER_LIST)}
        return sorted(models, key=lambda m: (idx.get(m, 10_000), m))

    MODELS = order_models(sorted(agg['model_for_match'].dropna().unique().tolist()))
    ALPHAS = np.array(ALPHAS_FIXED, dtype=float)
    _disp = globals().get('_disp_label', lambda s: s.lstrip('_'))
    _legend_ncols = min(globals().get('LEGEND_NCOLS', 4), max(1, len(MODELS)))

    def series_dict(col: str):
        d = {}
        for m in MODELS:
            sub = (agg[agg['model_for_match'] == m]
                   .set_index('alpha')
                   .sort_index()
                   .reindex(ALPHAS))
            d[m] = sub[col].tolist()
        return d

    metrics = [
        ('median_branches_pred', 'branches number (median)', '(a) branches number (median)', False),
        ('sat_prob_pred',        'Phase Probability (SAT pred)',      '(b) Phase probability', True),
        ('accuracy',             'Accuracy',                           '(c) Accuracy',          True),
        ('precision_sat',        'Precision (SAT)',        '(d) Precision (SAT as Positive)',  True),
        ('recall_sat',           'Recall (SAT)',           '(e) Recall (SAT as Positive)',     True),
        ('f1_sat',               'F1 (SAT)',               '(f) F1 (SAT as Positive)',         True),
        ('precision_unsat',      'Precision (UNSAT)',      '(g) Precision (UNSAT as Positive)',True),
        ('recall_unsat',         'Recall (UNSAT)',         '(h) Recall (UNSAT as Positive)',   True),
        ('f1_unsat',             'F1 (UNSAT)',             '(i) F1 (UNSAT as Positive)',       True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(24, 16))
    axes = axes.ravel()

    for ax, (col, ylabel, subtitle, unit_01) in zip(axes, metrics):
        dct = series_dict(col)
        for i, m in enumerate(MODELS):
            y = dct[m]
            st = get_style(m, fallback_idx=i)
            ax.plot(
                ALPHAS, y,
                color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
                label=_disp(m)
            )
        ax.set_xlabel('L / N (alpha)')
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        if unit_01:
            ax.set_ylim(0, 1.05)
        ax.set_xlim(float(np.min(ALPHAS)), float(np.max(ALPHAS)))
        ax.set_xticks(ALPHAS)
        ax.grid(True, linewidth=1.2)

    handles, labels = [], []
    for i, m in enumerate(MODELS):
        st = get_style(m, fallback_idx=i)
        h = plt.Line2D([0], [0],
                       color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                       markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
                       label=_disp(m))
        handles.append(h); labels.append(_disp(m))
    fig.legend(handles, labels, loc='lower center',
               ncol=_legend_ncols, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(wspace=0.36, hspace=0.48, bottom=0.18)

    for ext in ('png', 'pdf', 'svg'):
        out_path = os.path.join(out_dir, f'{file_stub}.{ext}')
        fig.savefig(out_path, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        print(f'[OK] 图已保存: {out_path}')
    plt.close(fig)

# ------------------------ 新增：单独绘制 MCC ------------------------
def plot_mcc(agg: pd.DataFrame, out_dir: str, ALPHAS_FIXED, file_stub: str = 'combined_metrics_mcc'):
    # 与 3x3 相同的样式/顺序
    CLAUDE_MODELS        = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus", "claude-sonnet-4"]
    DEEPSEEK_MODELS      = ["deepseek-chat", "deepseek-reasoner"]
    OPENAI_SPECIAL_RED   = ["gpt-5", "o1", "o3-mini"]
    OPENAI_OTHER_YELLOW  = ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]
    ORDER_LIST = CLAUDE_MODELS + DEEPSEEK_MODELS + OPENAI_SPECIAL_RED + OPENAI_OTHER_YELLOW + ["CDCL"]

    FAMILY_STYLE = {
        "claude"     : dict(color="#7b2cbf", linestyle="--"),
        "deepseek"   : dict(color="#2ca02c", linestyle=(0, (10, 6))),
        "openai_red" : dict(color="#d62728", linestyle="-"),
        "openai_yel" : dict(color="#ffbf00", linestyle="-"),
    }
    CLAUDE_TRIANGLES  = ["^", "v", "<", ">"]
    DEEPSEEK_MARKERS  = ["*", "s"]
    OPENAI_RED_MARKS  = ["o", "D", "X"]
    OPENAI_YEL_MARKS  = ["P", "h", "d"]

    def _which_openai_palette(name_lower: str) -> str:
        return "openai_red" if any(k in name_lower for k in ["gpt-5", "o1", "o3-mini"]) else "openai_yel"

    def get_style(model_name: str, fallback_idx: int = 0):
        name = (model_name or "").lower()
        if name == "cdcl":
            return {"color": "black", "linestyle": "-", "marker": "*"}
        if "claude" in name:
            fam, markers = "claude", CLAUDE_TRIANGLES
        elif "deepseek" in name:
            fam, markers = "deepseek", DEEPSEEK_MARKERS
        elif ("gpt" in name) or ("o" in name) or ("openai" in name):
            fam = _which_openai_palette(name)
            markers = OPENAI_RED_MARKS if fam == "openai_red" else OPENAI_YEL_MARKS
        else:
            fam, markers = "openai_yel", OPENAI_YEL_MARKS
        style = {**FAMILY_STYLE[fam]}
        style["marker"] = markers[fallback_idx % len(markers)]
        return style

    def order_models(models: list) -> list:
        idx = {m: i for i, m in enumerate(ORDER_LIST)}
        return sorted(models, key=lambda m: (idx.get(m, 10_000), m))

    MODELS = order_models(sorted(agg['model_for_match'].dropna().unique().tolist()))
    ALPHAS = np.array(ALPHAS_FIXED, dtype=float)
    _disp = globals().get('_disp_label', lambda s: s.lstrip('_'))
    _legend_ncols = min(globals().get('LEGEND_NCOLS', 4), max(1, len(MODELS)))

    # 准备 mcc 序列
    dct = {}
    for m in MODELS:
        sub = (agg[agg['model_for_match'] == m]
               .set_index('alpha')
               .sort_index()
               .reindex(ALPHAS))
        dct[m] = sub['mcc'].tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(MODELS):
        y = dct[m]
        st = get_style(m, fallback_idx=i)
        ax.plot(
            ALPHAS, y,
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
            label=_disp(m)
        )

    ax.set_xlabel('L / N (alpha)')
    ax.set_ylabel('MCC')
    ax.set_title('Matthews Correlation Coefficient (MCC)')
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(float(np.min(ALPHAS)), float(np.max(ALPHAS)))
    ax.set_xticks(ALPHAS)
    ax.grid(True, linewidth=1.2)

    handles, labels = [], []
    for i, m in enumerate(MODELS):
        st = get_style(m, fallback_idx=i)
        h = plt.Line2D([0], [0],
                       color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                       markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
                       label=_disp(m))
        handles.append(h); labels.append(_disp(m))
    fig.legend(handles, labels, loc='lower center',
               ncol=_legend_ncols, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(bottom=0.22)

    for ext in ('png', 'pdf', 'svg'):
        out_path = os.path.join(out_dir, f'{file_stub}.{ext}')
        fig.savefig(out_path, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        print(f'[OK] MCC 图已保存: {out_path}')
    plt.close(fig)

# ------------------------ 主流程 ------------------------
def main():
    df = combine_datasets([ONE_DATASET, SECOND_DATASET])
    agg = aggregate_by_model_alpha(df)

    # 导出聚合表（包含 mcc）
    xlsx_path = os.path.join(OUT_DIR, 'combined_metrics_by_model_alpha.xlsx')
    with pd.ExcelWriter(xlsx_path) as w:
        agg.to_excel(w, index=False, sheet_name='metrics')
    print(f'[OK] 指标已写入: {xlsx_path}')

    # 画 3×3（原有）
    plot_3x3(agg, OUT_DIR, ALPHAS_FIXED, file_stub='combined_metrics_3x3')

    # 画 MCC（单独一张图）
    plot_mcc(agg, OUT_DIR, ALPHAS_FIXED, file_stub='combined_metrics_mcc')

    # 画 4×3（第 4 行放 MCC + 两块 legend）
    plot_4x3_with_mcc_and_legends(agg, OUT_DIR, ALPHAS_FIXED, file_stub='combined_metrics_4x3')



def plot_4x3_with_mcc_and_legends(agg: pd.DataFrame, out_dir: str, ALPHAS_FIXED, file_stub: str = 'combined_metrics_4x3'):
    import math

    # 与 3x3 相同的样式/顺序
    CLAUDE_MODELS        = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus", "claude-sonnet-4"]
    DEEPSEEK_MODELS      = ["deepseek-chat", "deepseek-reasoner"]
    OPENAI_SPECIAL_RED   = ["gpt-5", "o1", "o3-mini"]
    OPENAI_OTHER_YELLOW  = ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]
    ORDER_LIST = CLAUDE_MODELS + DEEPSEEK_MODELS + OPENAI_SPECIAL_RED + OPENAI_OTHER_YELLOW + ["CDCL"]

    FAMILY_STYLE = {
        "claude"     : dict(color="#7b2cbf", linestyle="--"),
        "deepseek"   : dict(color="#2ca02c", linestyle=(0, (10, 6))),
        "openai_red" : dict(color="#d62728", linestyle="-"),
        "openai_yel" : dict(color="#ffbf00", linestyle="-"),
    }
    CLAUDE_TRIANGLES  = ["^", "v", "<", ">"]
    DEEPSEEK_MARKERS  = ["*", "s"]
    OPENAI_RED_MARKS  = ["o", "D", "X"]
    OPENAI_YEL_MARKS  = ["P", "h", "d"]

    def _which_openai_palette(name_lower: str) -> str:
        return "openai_red" if any(k in name_lower for k in ["gpt-5", "o1", "o3-mini"]) else "openai_yel"

    def get_style(model_name: str, fallback_idx: int = 0):
        name = (model_name or "").lower()
        if name == "cdcl":
            return {"color": "black", "linestyle": "-", "marker": "*"}
        if "claude" in name:
            fam, markers = "claude", CLAUDE_TRIANGLES
        elif "deepseek" in name:
            fam, markers = "deepseek", DEEPSEEK_MARKERS
        elif ("gpt" in name) or ("o" in name) or ("openai" in name):
            fam = _which_openai_palette(name)
            markers = OPENAI_RED_MARKS if fam == "openai_red" else OPENAI_YEL_MARKS
        else:
            fam, markers = "openai_yel", OPENAI_YEL_MARKS
        style = {**FAMILY_STYLE[fam]}
        style["marker"] = markers[fallback_idx % len(markers)]
        return style

    def order_models(models: list) -> list:
        idx = {m: i for i, m in enumerate(ORDER_LIST)}
        return sorted(models, key=lambda m: (idx.get(m, 10_000), m))

    MODELS = order_models(sorted(agg['model_for_match'].dropna().unique().tolist()))
    ALPHAS = np.array(ALPHAS_FIXED, dtype=float)
    _disp = globals().get('_disp_label', lambda s: s.lstrip('_'))
    _legend_ncols = min(globals().get('LEGEND_NCOLS', 2), max(1, len(MODELS)))

    # 9 个原指标（保持与你的 3×3 一致）
    metrics = [
        ('median_branches_pred', 'branches number (median)', '(a) branches number (median)', False),
        ('sat_prob_pred',        'Phase Probability (SAT pred)',      '(b) Phase probability', True),
        ('accuracy',             'Accuracy',                           '(c) Accuracy',          True),
        ('precision_sat',        'Precision (SAT)',        '(d) Precision (SAT as Positive)',  True),
        ('recall_sat',           'Recall (SAT)',           '(e) Recall (SAT as Positive)',     True),
        ('f1_sat',               'F1 (SAT)',               '(f) F1 (SAT as Positive)',         True),
        ('precision_unsat',      'Precision (UNSAT)',      '(g) Precision (UNSAT as Positive)',True),
        ('recall_unsat',         'Recall (UNSAT)',         '(h) Recall (UNSAT as Positive)',   True),
        ('f1_unsat',             'F1 (UNSAT)',             '(i) F1 (UNSAT as Positive)',       True),
    ]

    # 准备一个便捷函数：给定列名 -> 每个模型的 y 序列（按固定 alpha 对齐）
    def series_dict(col: str):
        d = {}
        for m in MODELS:
            sub = (agg[agg['model_for_match'] == m]
                   .set_index('alpha')
                   .sort_index()
                   .reindex(ALPHAS))
            d[m] = sub[col].tolist()
        return d

    # 开始画 4×3
    fig, axes = plt.subplots(4, 3, figsize=(24, 22))
    axes = axes.ravel()

    # 前 9 个子图：沿用 3×3 的逻辑
    for idx, (col, ylabel, subtitle, unit_01) in enumerate(metrics):
        ax = axes[idx]
        dct = series_dict(col)
        for i, m in enumerate(MODELS):
            y = dct[m]
            st = get_style(m, fallback_idx=i)
            ax.plot(
                ALPHAS, y,
                color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
                label=_disp(m)
            )
        ax.set_xlabel('L / N (alpha)')
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        if unit_01:
            ax.set_ylim(0, 1.05)
        ax.set_xlim(float(np.min(ALPHAS)), float(np.max(ALPHAS)))
        ax.set_xticks(ALPHAS)
        ax.grid(True, linewidth=1.2)

    # 第 10 个子图（索引 9）：MCC
    ax_mcc = axes[9]
    dct_mcc = series_dict('mcc')
    for i, m in enumerate(MODELS):
        y = dct_mcc[m]
        st = get_style(m, fallback_idx=i)
        ax_mcc.plot(
            ALPHAS, y,
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
            label=_disp(m)
        )
    ax_mcc.set_xlabel('L / N (alpha)')
    ax_mcc.set_ylabel('MCC')
    ax_mcc.set_title('(j) Matthews Correlation Coefficient (MCC)')
    ax_mcc.set_ylim(-1.05, 1.05)
    ax_mcc.set_xlim(float(np.min(ALPHAS)), float(np.max(ALPHAS)))
    ax_mcc.set_xticks(ALPHAS)
    ax_mcc.grid(True, linewidth=1.2)

    # 第 11、12 个子图（索引 10, 11）：legend 面板
    handles, labels = [], []
    for i, m in enumerate(MODELS):
        st = get_style(m, fallback_idx=i)
        h = plt.Line2D([0], [0],
                       color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                       markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
                       label=_disp(m))
        handles.append(h); labels.append(_disp(m))

    # 拆成两半，分别放在 (4,2) 与 (4,3)
    half = math.ceil(len(handles) / 2)
    ax_leg_left  = axes[10]
    ax_leg_right = axes[11]
    ax_leg_left.axis('off')
    ax_leg_right.axis('off')

    ax_leg_left.legend(handles=handles, labels=labels,
                       loc='center', ncol=2, frameon=True,
                       columnspacing=1.4, handletextpad=0.6)
    # ax_leg_right.legend(handles=handles[half:], labels=labels[half:],
    #                     loc='center', ncol=2, frameon=True,
    #                     columnspacing=1.4, handletextpad=0.6)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.48, wspace=0.36)

    for ext in ('png', 'pdf', 'svg'):
        out_path = os.path.join(out_dir, f'{file_stub}.{ext}')
        fig.savefig(out_path, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        print(f'[OK] 4x3 图已保存: {out_path}')
    plt.close(fig)






if __name__ == '__main__':
    main()
