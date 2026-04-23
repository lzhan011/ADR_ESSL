import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from draw_legend import *
base_dir = os.path.dirname(__file__)
# ------------------------ 配置路径 ------------------------
# INTERMEDIATE_XLSX = (
#     r'C:\Research\Vulnerability\Satisfiability_Solvers\Figure_in_paper\LLMs_phase_transition\combined_metrics\combined_metrics_by_model_alpha.xlsx'
# )
INTERMEDIATE_XLSX = (os.path.join(base_dir, '..', 'Analysis_Result_Collection', 'Figure_in_paper',
    'LLMs_phase_transition','combined_metrics','combined_metrics_by_model_alpha.xlsx')
)
OUT_DIR = os.path.dirname(INTERMEDIATE_XLSX)
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
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 2,
        "figure.titlesize": base_font + 2,
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


def _disp_label(name: str) -> str:
    return name.lstrip('_')


def load_aggregated_metrics_xlsx(path: str) -> pd.DataFrame:
    agg = pd.read_excel(path, sheet_name='metrics')

    needed = {
        'model_for_match',
        'alpha',
        'median_branches_pred',
        'sat_prob_pred',
        'accuracy',
        'precision_sat',
        'recall_sat',
        'f1_sat',
        'precision_unsat',
        'recall_unsat',
        'f1_unsat',
        'mcc',
    }
    missing = needed - set(agg.columns)
    if missing:
        raise ValueError(f'中间文件缺少必要列: {missing} in {path}')

    agg = agg.copy()
    agg['model_for_match'] = agg['model_for_match'].astype(str)
    agg['alpha'] = pd.to_numeric(agg['alpha'], errors='coerce').round(1)

    numeric_cols = [
        'median_branches_pred',
        'sat_prob_pred',
        'accuracy',
        'precision_sat',
        'recall_sat',
        'f1_sat',
        'precision_unsat',
        'recall_unsat',
        'f1_unsat',
        'mcc',
    ]
    for col in numeric_cols:
        agg[col] = pd.to_numeric(agg[col], errors='coerce')

    return agg


def plot_4x3_with_mcc_and_legends(
    agg: pd.DataFrame,
    out_dir: str,
    ALPHAS_FIXED,
    file_stub: str = 'combined_metrics_4x3'
):
    import math

    # 与原始脚本保持一致的样式/顺序
    CLAUDE_MODELS = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus", "claude-sonnet-4"]
    DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
    OPENAI_SPECIAL_RED = ["gpt-5", "o1", "o3-mini"]
    OPENAI_OTHER_YELLOW = ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]
    ORDER_LIST = CLAUDE_MODELS + DEEPSEEK_MODELS + OPENAI_SPECIAL_RED + OPENAI_OTHER_YELLOW + ["CDCL"]

    FAMILY_STYLE = {
        "claude": dict(color="#7b2cbf", linestyle="--"),
        "deepseek": dict(color="#2ca02c", linestyle=(0, (10, 6))),
        "openai_red": dict(color="#d62728", linestyle="-"),
        "openai_yel": dict(color="#ffbf00", linestyle="-"),
    }
    CLAUDE_TRIANGLES = ["^", "v", "<", ">"]
    DEEPSEEK_MARKERS = ["*", "s"]
    OPENAI_RED_MARKS = ["o", "D", "X"]
    OPENAI_YEL_MARKS = ["P", "h", "d"]

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

    metrics = [
        ('median_branches_pred', 'branches number (median)', '(a) branches number (median)', False),
        ('sat_prob_pred', 'Phase Probability (SAT pred)', '(b) Phase probability', True),
        ('accuracy', 'Accuracy', '(c) Accuracy', True),
        ('precision_sat', 'Precision (SAT)', '(d) Precision (SAT as Positive)', True),
        ('recall_sat', 'Recall (SAT)', '(e) Recall (SAT as Positive)', True),
        ('f1_sat', 'F1 (SAT)', '(f) F1 (SAT as Positive)', True),
        ('precision_unsat', 'Precision (UNSAT)', '(g) Precision (UNSAT as Positive)', True),
        ('recall_unsat', 'Recall (UNSAT)', '(h) Recall (UNSAT as Positive)', True),
        ('f1_unsat', 'F1 (UNSAT)', '(i) F1 (UNSAT as Positive)', True),
    ]

    def series_dict(col: str):
        d = {}
        for m in MODELS:
            sub = (
                agg[agg['model_for_match'] == m]
                .set_index('alpha')
                .sort_index()
                .reindex(ALPHAS)
            )
            d[m] = sub[col].tolist()
        return d

    fig, axes = plt.subplots(4, 3, figsize=(24, 22))
    axes = axes.ravel()

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

    handles, labels = [], []
    for i, m in enumerate(MODELS):
        st = get_style(m, fallback_idx=i)
        h = plt.Line2D(
            [0], [0],
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="black",
            label=_disp(m)
        )
        handles.append(h)
        labels.append(_disp(m))

    half = math.ceil(len(handles) / 2)
    ax_leg_left = axes[10]
    ax_leg_right = axes[11]
    ax_leg_left.axis('off')
    ax_leg_right.axis('off')

    ax_leg_left.legend(
        handles=handles,
        labels=labels,
        loc='center',
        ncol=2,
        frameon=True,
        columnspacing=1.4,
        handletextpad=0.6
    )
    # ax_leg_right.legend(handles=handles[half:], labels=labels[half:],
    #                     loc='center', ncol=2, frameon=True,
    #                     columnspacing=1.4, handletextpad=0.6)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.48, wspace=0.36)

    for ext in ('png', 'pdf', 'svg'):
        out_path = os.path.join(out_dir, f'{file_stub}.{ext}')
        fig.savefig(out_path, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        print(f'[OK] 4x3 Figure Saved: {out_path}')
    plt.close(fig)


def main():
    agg = load_aggregated_metrics_xlsx(INTERMEDIATE_XLSX)
    plot_4x3_with_mcc_and_legends(agg, OUT_DIR, ALPHAS_FIXED, file_stub='combined_metrics_4x3')


if __name__ == '__main__':
    main()
