import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
# ========= 配置路径 =========
# ========= Path configuration =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)   # Go up from SAT_Draw_Figure to Code root
BASE_DIR = os.path.join(CODE_DIR, "Analysis_Result_Collection")

xls_path = os.path.join(BASE_DIR, "SAT_figures.xlsx")
save_dir = os.path.join(BASE_DIR, "Figure_in_paper", "pairs")
os.makedirs(save_dir, exist_ok=True)
save_base = os.path.join(save_dir, "pairs_small_alpha_prediction_correct_rate")

print("Script directory:", SCRIPT_DIR)
print("Code root directory:", CODE_DIR)
print("Input Excel path:", xls_path)
print("Output directory:", save_dir)
print("Output base path:", save_base)
# ========= 读 Fixed_Paired =========
df_SAT_UNSAT = pd.read_excel(xls_path, sheet_name="Fixed_Paired", skiprows=2)
df_SAT_UNSAT = df_SAT_UNSAT[~((df_SAT_UNSAT["Model"] == "gpt-5") & (df_SAT_UNSAT["N"] == 50))]

need_cols4 = ["Model", "N", "Accuracy", "Precision", "Recall", "F1-score"]
df4 = df_SAT_UNSAT.dropna(subset=need_cols4)[need_cols4].copy()
df4["N"] = pd.to_numeric(df4["N"], errors="coerce")
for c in ["Accuracy", "Precision", "Recall", "F1-score"]:
    df4[c] = pd.to_numeric(df4[c], errors="coerce")
df4 = df4.dropna(subset=["N"])

# ========= 读 UNSAT 分组（用于第 2 行三个子图）=========
UNSAT_need_cols = ['Model', 'N.1', 'Accuracy.1', 'Precision.1', 'Recall.1', 'F1-score.1']
df_UNSAT = df_SAT_UNSAT.dropna(subset=UNSAT_need_cols)[UNSAT_need_cols].copy()
df_UNSAT = df_UNSAT.rename(columns={
    'N.1': "N",
    'Accuracy.1': 'Accuracy',
    'Precision.1': 'Precision',
    'Recall.1': 'Recall',
    'F1-score.1': 'F1-score'
}).copy()

df_UNSAT["N"] = pd.to_numeric(df_UNSAT["N"], errors="coerce")
for c in ["Accuracy", "Precision", "Recall", "F1-score"]:
    df_UNSAT[c] = pd.to_numeric(df_UNSAT[c], errors="coerce")
df_UNSAT = df_UNSAT.dropna(subset=["N"])

# ========= 读 ADR + MCC =========
df_adr = pd.read_excel(xls_path, sheet_name="Fixed_Paired_Our_New_Metrics", skiprows=0)

df_adr = df_adr.rename(columns={
    "model_select": "Model",
    "ADR (Accurate Differentiation Rate)": "ADR"
})
df_adr = df_adr[~((df_adr["Model"] == "gpt-5") & (df_adr["N"] == 50))]

# 找 ADR 列
adr_col_candidates = [c for c in df_adr.columns if "ADR" in str(c)]
if not adr_col_candidates:
    raise ValueError("未在 sheet 'Fixed_Paired_Our_New_Metrics' 中找到 ADR 列。")
adr_col = "ADR"

# 找 MCC 列
mcc_candidates = [c for c in df_adr.columns if str(c).strip().upper() == "MCC" or "MCC" in str(c)]
if not mcc_candidates:
    raise ValueError("未在 sheet 'Fixed_Paired_Our_New_Metrics' 中找到 MCC 列。")
mcc_col = mcc_candidates[0]

# 整理 ADR 与 MCC 两个 DataFrame
_df_common = df_adr.dropna(subset=["Model", "N"]).copy()
_df_common["N"] = pd.to_numeric(_df_common["N"], errors="coerce")
_df_common = _df_common.dropna(subset=["N"])

df_adr_clean = _df_common.dropna(subset=[adr_col])[["Model", "N", adr_col]].copy()
df_adr_clean[adr_col] = pd.to_numeric(df_adr_clean[adr_col], errors="coerce")

df_mcc_clean = _df_common.dropna(subset=[mcc_col])[["Model", "N", mcc_col]].copy()
df_mcc_clean[mcc_col] = pd.to_numeric(df_mcc_clean[mcc_col], errors="coerce")

# ========= 新增：读取 Assignments_satisfied_rate =========
# 允许 model 列既可能叫 "model_name" 也可能叫 "Model"（大小写不敏感）
df_assign_raw = pd.read_excel(xls_path, sheet_name="Assignments_satisfied_rate",skiprows=2)
cols_lower = {c.lower(): c for c in df_assign_raw.columns}

model_col = None
for cand in ["model", "model_name"]:
    if cand in cols_lower:
        model_col = cols_lower[cand]
        break
if model_col is None:
    raise ValueError("在 sheet 'Assignments_satisfied_rate' 中未找到 model_name / Model 列。")

# N 列
if "n" in cols_lower:
    n_col = cols_lower["n"]
else:
    raise ValueError("在 sheet 'Assignments_satisfied_rate' 中未找到 N 列。")

# rate 列（大小写不敏感匹配，允许包含）
rate_col = None
for c in df_assign_raw.columns:
    if str(c).strip().lower() == "assignments_satisfied_rate":
        rate_col = c
        break
if rate_col is None:
    # 容错：如果列名里包含该关键词也接受
    cand = [c for c in df_assign_raw.columns if "assignments" in str(c).lower() and "satisfied" in str(c).lower()]
    if not cand:
        raise ValueError("在 sheet 'Assignments_satisfied_rate' 中未找到 Assignments_satisfied_rate 列。")
    rate_col = cand[0]

df_assign = df_assign_raw[[model_col, n_col, rate_col]].copy()
df_assign = df_assign.rename(columns={model_col: "Model", n_col: "N", rate_col: "Assignments_satisfied_rate"})
df_assign["N"] = pd.to_numeric(df_assign["N"], errors="coerce")
df_assign["Assignments_satisfied_rate"] = pd.to_numeric(df_assign["Assignments_satisfied_rate"], errors="coerce")
df_assign = df_assign.dropna(subset=["Model", "N", "Assignments_satisfied_rate"])
df_assign = df_assign[~((df_assign["Model"] == "gpt-5") & (df_assign["N"] == 50))]

# ========= 仅保留白名单模型 =========
model_list = [
    'gpt-3.5-turbo-0125', 'gpt-4o-latest', 'gpt-4.1',
    'o3-mini', 'o1', 'gpt-5', 'deepseek-reasoner'
]
df4          = df4[df4['Model'].isin(model_list)].copy()
df_UNSAT     = df_UNSAT[df_UNSAT['Model'].isin(model_list)].copy()
df_adr_clean = df_adr_clean[df_adr_clean['Model'].isin(model_list)].copy()
df_mcc_clean = df_mcc_clean[df_mcc_clean['Model'].isin(model_list)].copy()
df_assign    = df_assign[df_assign['Model'].isin(model_list)].copy()

# ========= Legend 文本清理 =========
def legend_label(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    if low.startswith("claude"):
        s = re.sub(r"[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}$", "", s)
        s = re.sub(r"[-_]?20\d{6,8}$", "", s)
    if "deepseek" in low:
        s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s

# ========= 家族/样式 =========
COLOR_CLAUDE   = "#7b3fc8"   # 紫
COLOR_DEEPSEEK = "#2ca02c"   # 绿
COLOR_OAI_RED  = "#d62728"   # 红
COLOR_OAI_YELL = "#ffb000"   # 黄

LS_CLAUDE   = "--"
LS_DEEPSEEK = (0, (10, 4))
LS_OPENAI   = "-"

DEEPSEEK_SET = {"deepseek-chat": "*", "deepseek-reasoner": "s"}
OPENAI_RED_SET = {"gpt-5": "o", "o1": "D", "o3-mini": "X"}
OPENAI_YELLOW_SET = {"gpt-4.1": "P", "gpt-3.5-turbo-0125": "d", "gpt-4o-latest": "h"}

def _claude_marker(ml: str) -> str:
    ml = ml.lower()
    if "haiku" in ml: return "^"
    if "sonnet" in ml and (("3-7" in ml) or ("3.7" in ml)): return "v"
    if ("3-opus" in ml) or (("opus" in ml) and re.search(r"\b3([._-]\d+)?\b", ml)): return "<"
    if re.search(r"(opus|sonnet)[-_ ]?4\b", ml) or "opus-4" in ml: return ">"
    return ["^","v","<",">"][hash(ml) % 4]

def _openai_yellow_marker(ml: str) -> str:
    ml = ml.lower()
    if "gpt-4o-latest" in ml: return "h"
    if re.search(r"\bgpt[-_]?4[.\-]?1\b", ml): return "P"
    if "gpt-3.5-turbo-0125" in ml: return "d"
    return ["P", "d", "h"][hash(ml) % 3]

def _in_whitelist(ml: str) -> bool:
    ml = ml.lower()
    if ml.startswith("claude"): return True
    if "deepseek" in ml: return True
    if ml in OPENAI_RED_SET: return True
    if any(p in ml for p in ["gpt-4.1", "gpt-3.5-turbo-0125", "gpt-4o-latest"]):
        return True
    return False

def style_for(model: str):
    m  = str(model).strip()
    ml = m.lower()
    if not _in_whitelist(ml):
        return None
    if ml.startswith("claude"):
        return ("claude", COLOR_CLAUDE, LS_CLAUDE, _claude_marker(ml))
    if ml in DEEPSEEK_SET or "deepseek" in ml:
        mk = DEEPSEEK_SET.get(ml, "*")
        return ("deepseek", COLOR_DEEPSEEK, LS_DEEPSEEK, mk)
    if ml in OPENAI_RED_SET:
        return ("openai-red", COLOR_OAI_RED, LS_OPENAI, OPENAI_RED_SET[ml])
    mk = OPENAI_YELLOW_SET.get(ml, _openai_yellow_marker(ml))
    return ("openai-yellow", COLOR_OAI_YELL, LS_OPENAI, mk)

# ========= 全局样式 =========
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 26, "axes.titlesize": 20, "axes.labelsize": 16,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 24,
    "lines.linewidth": 3.0, "lines.markersize": 10.0,
    "axes.linewidth": 1.5, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 6, "ytick.major.size": 6, "savefig.dpi": 600,
})

# ========= 等距刻度设置 =========
X_ORDER = [5, 8, 10, 25, 50, 60, 75]
POS_MAP = {v: i for i, v in enumerate(X_ORDER)}

def _prep_positions(d: pd.DataFrame) -> pd.DataFrame:
    dd = d[d["N"].isin(X_ORDER)].copy()
    dd["N_pos"] = dd["N"].map(POS_MAP).astype(int)
    return dd

df4          = _prep_positions(df4)          # SAT 指标
df_UNSAT     = _prep_positions(df_UNSAT)     # UNSAT 指标
df_adr_clean = _prep_positions(df_adr_clean)
df_mcc_clean = _prep_positions(df_mcc_clean)
df_assign    = _prep_positions(df_assign)    # Assignments_satisfied_rate

# ========= 生成绘制顺序 =========
def ordered_models(series_models: pd.Series) -> list:
    uniq = [str(x) for x in series_models.dropna().unique().tolist()]
    groups = {"claude":[], "deepseek":[], "openai-red":[], "openai-yellow":[], "_drop":[]}
    for m in uniq:
        st = style_for(m)
        if st is None:
            groups["_drop"].append(m)
        else:
            g, *_ = st
            groups[g].append(m)
    return groups["claude"] + groups["deepseek"] + groups["openai-red"] + groups["openai-yellow"]

# 将 UNSAT、ADR、MCC、Assignments 也纳入模型集合
models_all = pd.Series(pd.concat([
    df4["Model"], df_UNSAT["Model"], df_adr_clean["Model"], df_mcc_clean["Model"], df_assign["Model"]
], ignore_index=True))
MODELS = ordered_models(models_all)

# ========= 4×3 画布（12 子图）=========
fig, axes = plt.subplots(4, 3, figsize=(20, 18))
axes = axes.ravel()

PANEL_TAGS = list("abcdefghijkl")  # 12 个：a..l

# ======= 重新排布 12 个子图的顺序 =======
# 第 1 行（SAT）：Precision, Recall, F1
# 第 2 行（UNSAT）：Precision, Recall, F1
# 第 3 行：Accuracy, MCC, ADR
# 第 4 行：Assignments_satisfied_rate, (empty), (empty)
panels = [
    # Row 1: SAT
    ("Precision (SAT)",  "Precision",  df4),
    ("Recall (SAT)",     "Recall",     df4),
    ("F1-score (SAT)",   "F1-score",   df4),

    # Row 2: UNSAT
    ("Precision (UNSAT)", "Precision",  df_UNSAT),
    ("Recall (UNSAT)",    "Recall",     df_UNSAT),
    ("F1-score (UNSAT)",  "F1-score",   df_UNSAT),

    # Row 3: Overall Accuracy + Our metrics
    ("Accuracy",          "Accuracy",   df4),
    ("MCC",               mcc_col,      df_mcc_clean),
    ("ADR",               adr_col,      df_adr_clean),

    # Row 4: Assignments_satisfied_rate + two empty placeholders
    ("Assignments_satisfied_rate", "Assignments_satisfied_rate", df_assign),
    (None, None, None),
    (None, None, None),
]

# 绘制 12 个子图
for idx, (title, col, dframe) in enumerate(panels):
    ax = axes[idx]

    if title is None:
        # 占位空白图
        ax.axis('off')
        ax.set_title(f"({PANEL_TAGS[idx]})", pad=10)
        continue

    for m in MODELS:
        st = style_for(m)
        if st is None:
            continue
        group, color, ls, marker = st
        sub = dframe[dframe["Model"].astype(str) == m].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("N_pos")
        ax.plot(
            sub["N_pos"], sub[col],
            label=legend_label(m),
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.4
        )

    ax.set_title(f"({PANEL_TAGS[idx]}) {title} vs N")
    ax.set_xlabel("N")
    # y 轴标签：取指标名（第一个空格前），或直接用列名（Assignments_satisfied_rate）
    if " " in title:
        ax.set_ylabel(title.split(" ")[0])
    else:
        ax.set_ylabel(title)

    # 0-1 指标范围（包括 Assignments_satisfied_rate 与 ADR/F1/Precision/Recall/Accuracy）
    if any(title.startswith(t) for t in ["Accuracy", "Precision", "Recall", "F1-score", "ADR", "Assignments_satisfied_rate"]):
        ax.set_ylim(-0.05, 1.05)
    # MCC 特殊范围 [-1, 1]
    if title.startswith("MCC"):
        ax.set_ylim(-1.05, 1.05)

    ax.set_xlim(-0.4, len(X_ORDER)-0.6)
    ax.set_xticks(range(len(X_ORDER)))
    ax.set_xticklabels([str(x) for x in X_ORDER])
    ax.grid(True, linewidth=1.0)

# ========= 底部 legend（3 列，多行自动换行）=========
handles, labels = [], []
for m in MODELS:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)
    h = Line2D([0], [0], color=color, linestyle=ls, marker=marker,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.4, label=lab)
    handles.append(h); labels.append(lab)

# 放到底部，3 列，自动分多行
fig.legend(handles=handles, labels=labels,
           loc='lower center', ncol=3, frameon=True,
           columnspacing=1.4, handletextpad=0.6, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.12, 1, 1])  # 给底部 legend 预留空间
fig.subplots_adjust(hspace=0.35, wspace=0.25)

# ========= 保存 =========
fig.savefig(save_base + ".pdf", bbox_inches="tight")
fig.savefig(save_base + ".svg", bbox_inches="tight")
fig.savefig(save_base + ".png", dpi=300, bbox_inches="tight")
print("Saved:", save_base + ".[pdf|svg|png]")







def build_merged_metrics_pairs(
    df4, df_UNSAT, df_mcc_clean, df_adr_clean, df_assign,
    mcc_col="MCC", adr_col="ADR",
    filter_models=None, filter_N=None,
    export_path=None
):
    """
    Merge all subplot sources (SAT df4, UNSAT df_UNSAT, MCC/ADR, Assignments_satisfied_rate)
    horizontally on (Model, N).

    Output columns (when可用):
      Model, N,
      precision_sat, recall_sat, f1_sat, accuracy,
      precision_unsat, recall_unsat, f1_unsat,
      MCC, ADR,
      Assignments_satisfied_rate

    Parameters
    ----------
    df4 : pd.DataFrame
        SAT metrics. Expects columns ['Model','N','Accuracy','Precision','Recall','F1-score'] (subset ok).
    df_UNSAT : pd.DataFrame
        UNSAT metrics. Expects columns ['Model','N','Accuracy','Precision','Recall','F1-score'] (subset ok).
        Only Precision/Recall/F1-score are taken and renamed with *_unsat suffix.
    df_mcc_clean : pd.DataFrame
        MCC metrics. Must contain ['Model','N', mcc_col].
    df_adr_clean : pd.DataFrame
        ADR metrics. Must contain ['Model','N', adr_col].
    df_assign : pd.DataFrame
        Assignment rate. Must contain ['Model','N','Assignments_satisfied_rate'].
    mcc_col : str
        The column name of MCC in df_mcc_clean.
    adr_col : str
        The column name of ADR in df_adr_clean.
    filter_models : list or None
        If given, keep only rows whose Model ∈ filter_models.
    filter_N : list or None
        If given, keep only rows whose N ∈ filter_N.
    export_path : str or None
        If provided, save the merged table to this Excel path.

    Returns
    -------
    pd.DataFrame
        Merged wide table aligned by (Model, N).
    """
    def _safe_pick(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        have = [c for c in cols if c in df.columns]
        if not have:
            return pd.DataFrame(columns=cols)
        out = df[have].copy()
        return out

    # ——— SAT (df4)
    sat_src = _safe_pick(df4, ["Model", "N", "Accuracy", "Precision", "Recall", "F1-score"])
    # 统一为数值
    for c in ["N", "Accuracy", "Precision", "Recall", "F1-score"]:
        if c in sat_src.columns:
            sat_src[c] = pd.to_numeric(sat_src[c], errors="coerce")
    sat_src = sat_src.dropna(subset=["Model", "N"])
    sat = sat_src.rename(columns={
        "Accuracy":  "accuracy",
        "Precision": "precision_sat",
        "Recall":    "recall_sat",
        "F1-score":  "f1_sat",
    })[["Model","N","accuracy","precision_sat","recall_sat","f1_sat"]]

    # ——— UNSAT (df_UNSAT)
    unsat_src = _safe_pick(df_UNSAT, ["Model", "N", "Precision", "Recall", "F1-score"])
    for c in ["N", "Precision", "Recall", "F1-score"]:
        if c in unsat_src.columns:
            unsat_src[c] = pd.to_numeric(unsat_src[c], errors="coerce")
    unsat_src = unsat_src.dropna(subset=["Model", "N"])
    unsat = unsat_src.rename(columns={
        "Precision": "precision_unsat",
        "Recall":    "recall_unsat",
        "F1-score":  "f1_unsat",
    })[["Model","N","precision_unsat","recall_unsat","f1_unsat"]]

    # ——— MCC
    mcc_src = _safe_pick(df_mcc_clean, ["Model", "N", mcc_col])
    for c in ["N", mcc_col]:
        if c in mcc_src.columns:
            mcc_src[c] = pd.to_numeric(mcc_src[c], errors="coerce")
    mcc_src = mcc_src.dropna(subset=["Model","N"])
    mcc = mcc_src.rename(columns={mcc_col: "MCC"})[["Model","N","MCC"]]

    # ——— ADR
    adr_src = _safe_pick(df_adr_clean, ["Model", "N", adr_col])
    for c in ["N", adr_col]:
        if c in adr_src.columns:
            adr_src[c] = pd.to_numeric(adr_src[c], errors="coerce")
    adr_src = adr_src.dropna(subset=["Model","N"])
    adr = adr_src.rename(columns={adr_col: "ADR"})[["Model","N","ADR"]]

    # ——— Assignments_satisfied_rate
    assign_src = _safe_pick(df_assign, ["Model", "N", "Assignments_satisfied_rate"])
    for c in ["N", "Assignments_satisfied_rate"]:
        if c in assign_src.columns:
            assign_src[c] = pd.to_numeric(assign_src[c], errors="coerce")
    assign_src = assign_src.dropna(subset=["Model","N"])
    assign_df = assign_src[["Model","N","Assignments_satisfied_rate"]]

    # ——— Outer-merge by (Model, N)
    merged = sat.merge(unsat,   on=["Model","N"], how="outer") \
                .merge(mcc,     on=["Model","N"], how="outer") \
                .merge(adr,     on=["Model","N"], how="outer") \
                .merge(assign_df, on=["Model","N"], how="outer")

    # 过滤（如需）
    if filter_models is not None:
        merged = merged[merged["Model"].isin(filter_models)]
    if filter_N is not None:
        merged = merged[merged["N"].isin(filter_N)]

    # 排序与重置索引
    merged = merged.sort_values(by=["Model","N"]).reset_index(drop=True)

    # 可选导出
    if export_path:
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        except Exception:
            pass
        merged.to_excel(export_path, index=False)

    return merged

# —— 使用示例（追加在你的脚本末尾，不改动现有代码）——
merged_pairs = build_merged_metrics_pairs(
    df4, df_UNSAT, df_mcc_clean, df_adr_clean, df_assign,
    mcc_col=mcc_col, adr_col=adr_col,
    filter_models=model_list,    # 可选：按你的白名单过滤
    filter_N=X_ORDER,            # 可选：只保留绘制用的 N
    export_path=save_base + "_merged_metrics.xlsx"
)
print("Merged metrics shape:", merged_pairs.shape)


















# ========= Additional figure: 3x3 only, without Assignments_satisfied_rate =========

fig_3x3_only, axes_3x3_only = plt.subplots(3, 3, figsize=(20, 14))
axes_3x3_only = axes_3x3_only.ravel()

PANEL_TAGS_3X3 = list("abcdefghi")

# Only keep the first 9 panels
panels_3x3_only = [
    # Row 1: SAT
    ("Precision (SAT)", "Precision", df4),
    ("Recall (SAT)", "Recall", df4),
    ("F1-score (SAT)", "F1-score", df4),

    # Row 2: UNSAT
    ("Precision (UNSAT)", "Precision", df_UNSAT),
    ("Recall (UNSAT)", "Recall", df_UNSAT),
    ("F1-score (UNSAT)", "F1-score", df_UNSAT),

    # Row 3: Accuracy + our metrics
    ("Accuracy", "Accuracy", df4),
    ("MCC", mcc_col, df_mcc_clean),
    ("ADR", adr_col, df_adr_clean),
]

# Draw the 9 subplots
for idx, (title, col, dframe) in enumerate(panels_3x3_only):
    ax = axes_3x3_only[idx]

    for m in MODELS:
        st = style_for(m)
        if st is None:
            continue
        _, color, ls, marker = st

        sub = dframe[dframe["Model"].astype(str) == m].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("N_pos")

        ax.plot(
            sub["N_pos"],
            sub[col],
            label=legend_label(m),
            color=color,
            linestyle=ls,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4
        )

    ax.set_title(f"({PANEL_TAGS_3X3[idx]}) {title} vs N")
    ax.set_xlabel("N")

    if " " in title:
        ax.set_ylabel(title.split(" ")[0])
    else:
        ax.set_ylabel(title)

    if any(title.startswith(t) for t in ["Accuracy", "Precision", "Recall", "F1-score", "ADR"]):
        ax.set_ylim(-0.05, 1.05)

    if title.startswith("MCC"):
        ax.set_ylim(-1.05, 1.05)

    ax.set_xlim(-0.4, len(X_ORDER) - 0.6)
    ax.set_xticks(range(len(X_ORDER)))
    ax.set_xticklabels([str(x) for x in X_ORDER])
    ax.grid(True, linewidth=1.0)

# Build legend again
handles_3x3_only, labels_3x3_only = [], []
for m in MODELS:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)

    h = Line2D(
        [0], [0],
        color=color,
        linestyle=ls,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.4,
        label=lab
    )
    handles_3x3_only.append(h)
    labels_3x3_only.append(lab)

# Move the legend upward
fig_3x3_only.legend(
    handles=handles_3x3_only,
    labels=labels_3x3_only,
    loc="lower center",
    ncol=3,
    frameon=True,
    columnspacing=1.4,
    handletextpad=0.6,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig_3x3_only.subplots_adjust(hspace=0.35, wspace=0.25)

# Save with a new file name
save_base_3x3_only = save_base + "_3x3_only"

pdf_path_3x3_only = save_base_3x3_only + ".pdf"
svg_path_3x3_only = save_base_3x3_only + ".svg"
png_path_3x3_only = save_base_3x3_only + ".png"

fig_3x3_only.savefig(pdf_path_3x3_only, bbox_inches="tight")
fig_3x3_only.savefig(svg_path_3x3_only, bbox_inches="tight")
fig_3x3_only.savefig(png_path_3x3_only, dpi=300, bbox_inches="tight")

print("Saved PDF (3x3 only):", pdf_path_3x3_only)
print("Saved SVG (3x3 only):", svg_path_3x3_only)
print("Saved PNG (3x3 only):", png_path_3x3_only)

