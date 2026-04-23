import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ========= Path configuration =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)   # Go up from SAT_Draw_Figure to Code root
BASE_DIR = os.path.join(CODE_DIR, "Analysis_Result_Collection")

xls_path = os.path.join(BASE_DIR, "SAT_figures.xlsx")
save_dir = os.path.join(BASE_DIR, "Figure_in_paper", "2SAT")
os.makedirs(save_dir, exist_ok=True)
save_base = os.path.join(save_dir, "2SAT")

print("Script directory:", SCRIPT_DIR)
print("Code root directory:", CODE_DIR)
print("Input Excel path:", xls_path)
print("Output directory:", save_dir)
print("Output base path:", save_base)

# ---------- Cleaning and styling utilities ----------
DATE_SUFFIX_RE = re.compile(r'[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}$')  # -2025-08-30 or _20250830

def legend_label(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    # Claude: remove date suffix
    if low.startswith("claude"):
        s = DATE_SUFFIX_RE.sub("", s)
        s = re.sub(r"[-_]?20\d{6,8}$", "", s)
    # DeepSeek: remove parentheses and their contents
    if "deepseek" in low:
        s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s

# Family colors
COLOR_CLAUDE   = "#7b3fc8"   # purple
COLOR_DEEPSEEK = "#2ca02c"   # green
COLOR_OAI_RED  = "#d62728"   # red
COLOR_OAI_YELL = "#ffb000"   # yellow

# Line styles
LS_CLAUDE   = "--"
LS_DEEPSEEK = (0, (10, 4))
LS_OPENAI   = "-"

# Fixed markers
DEEPSEEK_SET = {
    "deepseek-chat": "*",
    "deepseek-reasoner": "s",
}
OPENAI_RED_SET = {
    "gpt-5": "o",
    "o1": "D",
    "o3-mini": "X",
}
OPENAI_YELLOW_SET = {
    "gpt-4.1": "P",
    "gpt-3.5-turbo-0125": "d",
    "gpt-4o-latest": "h",
}

def _claude_marker(ml: str) -> str:
    ml = ml.lower()
    if "haiku" in ml:
        return "^"
    if "sonnet" in ml and (("3-7" in ml) or ("3.7" in ml)):
        return "v"
    if ("3-opus" in ml) or (("opus" in ml) and re.search(r"\b3([._-]\d+)?\b", ml)):
        return "<"
    if re.search(r"(opus|sonnet)[-_ ]?4\b", ml) or "opus-4" in ml:
        return ">"
    return ["^", "v", "<", ">"][hash(ml) % 4]

def _openai_yellow_marker(ml: str) -> str:
    ml = ml.lower()
    if "gpt-4o-latest" in ml:
        return "h"
    if re.search(r"\bgpt[-_]?4[.\-]?1\b", ml):
        return "P"
    if "gpt-3.5-turbo-0125" in ml:
        return "d"
    return ["P", "d", "h"][hash(ml) % 3]

# Allowed models (whitelist logic)
WHITELIST = {
    # "deepseek-chat",
    "deepseek-reasoner",
    "gpt-5",
    # "o1",
    # "o3-mini",
    # "gpt-4.1",
    # "gpt-3.5-turbo-0125",
    "gpt-4o-latest"
}

def _in_whitelist(ml: str) -> bool:
    ml = ml.lower()
    if ml not in WHITELIST:
        if "opus-4" in ml:
            pass
        else:
            return False
    if ml.startswith("claude"):
        return True
    if "deepseek" in ml:
        return True
    if ml in OPENAI_RED_SET:
        return True
    if any(p in ml for p in ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]):
        return True
    return False

def style_for(model: str):
    """Return (group, color, linestyle, marker). Return None if not in whitelist."""
    m = str(model).strip()
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

# ========= Read and clean data =========
df_SAT_UNSAT = pd.read_excel(xls_path, sheet_name="2SAT_Traditional_Metrics", skiprows=2)

SAT_cols = ["model", "N", "Accuracy", "Precision", "Recall", "F1-score"]
UNSAT_cols = ["model.1", "N.1", "Accuracy.1", "Precision.1", "Recall.1", "F1-score.1"]

df_SAT = df_SAT_UNSAT[SAT_cols].dropna(subset=SAT_cols).copy()
df_SAT = df_SAT.rename(columns={"ACC": "Accuracy", "F1": "F1-score"})

df_UNSAT = df_SAT_UNSAT[UNSAT_cols].copy()
df_UNSAT = df_UNSAT.rename(columns={
    "model.1": "model", "N.1": "N",
    "Accuracy.1": "Accuracy", "Precision.1": "Precision",
    "Recall.1": "Recall", "F1-score.1": "F1-score"
})

# Convert data types
df_SAT["N"] = pd.to_numeric(df_SAT["N"], errors="coerce")
for c in ["Accuracy", "Precision", "Recall", "F1-score"]:
    df_SAT[c] = pd.to_numeric(df_SAT[c], errors="coerce")
df_SAT = df_SAT.dropna(subset=["N"])

df_UNSAT["N"] = pd.to_numeric(df_UNSAT["N"], errors="coerce")
for c in ["Accuracy", "Precision", "Recall", "F1-score"]:
    df_UNSAT[c] = pd.to_numeric(df_UNSAT[c], errors="coerce")
df_UNSAT = df_UNSAT.dropna(subset=["N"])

# Read ADR and MCC
ADR_DF = pd.read_excel(xls_path, sheet_name="2SAT_Our_New_Metrics", skiprows=0)
ADR_DF = ADR_DF.rename(columns={"ADR (Accurate Differentiation Rate)": "ADR"})
ADR_cols_flat = ["ADR", "MCC"]
ADR_DF = ADR_DF.dropna(subset=["model", "N"] + ADR_cols_flat)[["model", "N"] + ADR_cols_flat].copy()
ADR_DF["N"] = pd.to_numeric(ADR_DF["N"], errors="coerce")
for c in ADR_cols_flat:
    ADR_DF[c] = pd.to_numeric(ADR_DF[c], errors="coerce")
ADR_DF = ADR_DF.dropna(subset=["N"])

# Read assignment evaluation data for the 10th subplot
ASSIGNMENT_DF = pd.read_excel(xls_path, sheet_name="2SAT_Assignments_Evaluation", skiprows=2)
ASSIGNMENT_DF = ASSIGNMENT_DF.rename(columns={
    "Assignments_satisfied_rate": "Assignments_Satisfied_Rate",
    "model_name": "model"
})
ASSIGNMENT_col = "Assignments_Satisfied_Rate"
ASSIGNMENT_DF = ASSIGNMENT_DF.dropna(subset=["model", "N", ASSIGNMENT_col])[["model", "N", ASSIGNMENT_col]].copy()
ASSIGNMENT_DF["N"] = pd.to_numeric(ASSIGNMENT_DF["N"], errors="coerce")
ASSIGNMENT_DF[ASSIGNMENT_col] = pd.to_numeric(ASSIGNMENT_DF[ASSIGNMENT_col], errors="coerce")
ASSIGNMENT_DF = ASSIGNMENT_DF.dropna(subset=["N"])

# Normalize model names and remove duplicates
for dframe_name in ("df_SAT", "df_UNSAT", "ADR_DF", "ASSIGNMENT_DF"):
    dframe = locals()[dframe_name]
    dframe["model_clean"] = dframe["model"].astype(str).str.strip()
    low = dframe["model_clean"].str.lower()
    mask = ~low.str.contains(r"\bgpt-4-turbo\b", na=False)
    mask &= (low != "gpt-4o")
    mask &= ~low.str.contains(r"\bchatgpt-4o-latest\b", na=False)
    dframe = dframe[mask].copy()
    dframe.loc[:, "model_clean"] = dframe["model_clean"].apply(legend_label)
    dframe = dframe.sort_values(["model_clean", "N"])
    dframe = dframe.drop_duplicates(subset=["model_clean", "N"], keep="last")
# Keep only whitelist models
def _apply_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model_clean"].str.lower().map(_in_whitelist)].copy()

df_SAT = _apply_whitelist(df_SAT)
df_UNSAT = _apply_whitelist(df_UNSAT)
ADR_DF = _apply_whitelist(ADR_DF)
ASSIGNMENT_DF = _apply_whitelist(ASSIGNMENT_DF)

# Generate plotting order: Claude → DeepSeek → OpenAI-Red → OpenAI-Yellow
def ordered_models(series_models: pd.Series) -> list:
    uniq = [str(x) for x in series_models.dropna().unique().tolist()]
    groups = {"claude": [], "deepseek": [], "openai-red": [], "openai-yellow": [], "_drop": []}
    for m in uniq:
        st = style_for(m)
        if st is None:
            groups["_drop"].append(m)
        else:
            g, *_ = st
            groups[g].append(m)
    return groups["claude"] + groups["deepseek"] + groups["openai-red"] + groups["openai-yellow"]

models_all = pd.Series(pd.concat([
    df_SAT["model_clean"], df_UNSAT["model_clean"], ADR_DF["model_clean"], ASSIGNMENT_DF["model_clean"]
], ignore_index=True))
MODELS = ordered_models(models_all)

# ========= Global plotting style =========
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 28, "axes.titlesize": 24, "axes.labelsize": 28,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 20,
    "lines.linewidth": 3.0, "lines.markersize": 9,
})

# ========= Evenly spaced x-axis =========
x_order = [5, 8, 10, 25, 50]
xpos_map = {v: i for i, v in enumerate(x_order)}
xpos_ticks = list(range(len(x_order)))

# ========= Offset settings (used only for the 2nd subplot) =========
JITTER_RECALL_SAT_X = {"gpt-5": 0.0}
YJITTER_RECALL_SAT = {"gpt-5": -0.01}
JITTER_N_ONLY = {10, 25, 50}

# ========= Create 4×3 canvas (12 subplots) =========
fig, axes = plt.subplots(4, 3, figsize=(20, 18))
axes = axes.ravel()

# Panel order:
# First 9 panels follow the original layout;
# the 10th panel is the assignment metric;
# the 11th and 12th panels are left empty.
panels = [
    # Row 1: SAT -> precision, recall, F1
    ("Precision SAT", "Precision", df_SAT),
    ("Recall SAT", "Recall", df_SAT),
    ("F1 SAT", "F1-score", df_SAT),

    # Row 2: UNSAT -> precision, recall, F1
    ("Precision UNSAT", "Precision", df_UNSAT),
    ("Recall UNSAT", "Recall", df_UNSAT),
    ("F1 UNSAT", "F1-score", df_UNSAT),

    # Row 3: Accuracy, MCC, ADR
    ("Accuracy", "Accuracy", df_SAT),
    ("MCC", "MCC", ADR_DF),
    ("ADR", "ADR", ADR_DF),

    # Row 4: Assignments_Satisfied_Rate, (empty), (empty)
    ("Assignments_Satisfied_Rate", ASSIGNMENT_col, ASSIGNMENT_DF),
    (None, None, None),
    (None, None, None),
]

letters = list("abcdefghijkl")

for idx, (title, col, dframe) in enumerate(panels):
    ax = axes[idx]

    if title is None:
        ax.axis('off')
        ax.set_title(f"({letters[idx]})", pad=10)
        continue

    for m in MODELS:
        st = style_for(m)
        if st is None:
            continue
        group, color, ls, marker = st
        sub = dframe[dframe["model_clean"].str.lower() == m.lower()].copy()
        if sub.empty:
            continue
        sub = sub[sub["N"].isin(x_order)].copy()
        if sub.empty:
            continue
        sub["xpos"] = sub["N"].map(xpos_map)
        sub = sub.sort_values("xpos")

        xvals = sub["xpos"].astype(float).values
        yvals = sub[col].astype(float).values

        # Apply x/y jitter only for the 2nd subplot (Recall SAT, idx == 1)
        if idx == 1:
            xjit = JITTER_RECALL_SAT_X.get(m.lower(), 0.0)
            if xjit != 0.0:
                xvals = np.array([
                    xv + (xjit if x_order[int(xv)] in JITTER_N_ONLY else 0.0)
                    for xv in xvals
                ])
            yjit = YJITTER_RECALL_SAT.get(m.lower(), 0.0)
            if yjit != 0.0:
                xpos_list = sub["xpos"].astype(int).values
                yvals = np.array([
                    y + (yjit if x_order[xi] in JITTER_N_ONLY else 0.0)
                    for y, xi in zip(yvals, xpos_list)
                ])
                yvals = np.clip(yvals, 0.0, 1.0)

        ax.plot(
            xvals, yvals,
            label=legend_label(m),
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.4, clip_on=False
        )

    ax.set_title(f"({letters[idx]}) {title}")
    ax.set_xlabel("N")

    # Set y-axis label
    if title == "Assignments_Satisfied_Rate":
        ax.set_ylabel("Assignments_Satisfied_Rate")
    else:
        ax.set_ylabel(title.split()[0])

    ax.set_xticks(xpos_ticks)
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.3, len(x_order) - 0.7)

    if title == "MCC":
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#999999", lw=1.2)
    else:
        ax.set_ylim(-0.05, 1.05)

    ax.grid(True, linewidth=1)

# ========= Bottom legend =========
handles, labels = [], []
for m in MODELS:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)
    h = Line2D(
        [0], [0],
        color=color, linestyle=ls, marker=marker,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.4, label=lab
    )
    handles.append(h)
    labels.append(lab)

fig.legend(
    handles=handles, labels=labels,
    loc="lower center", ncol=4, frameon=True,
    columnspacing=1.4, handletextpad=0.6, bbox_to_anchor=(0.5, -0.02)
)

# ========= Layout and save =========
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.subplots_adjust(hspace=0.35, wspace=0.28)

os.makedirs(save_dir, exist_ok=True)

pdf_path = save_base + ".pdf"
svg_path = save_base + ".svg"
png_path = save_base + ".png"

fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
fig.savefig(svg_path, dpi=600, bbox_inches="tight")
fig.savefig(png_path, dpi=600, bbox_inches="tight")

print("Saved PDF:", pdf_path)
print("Saved SVG:", svg_path)
print("Saved PNG:", png_path)


def build_merged_metrics(df_SAT, df_UNSAT, ADR_DF, ASSIGNMENT_DF, assignment_col, export_path=None):
    """
    Horizontally merge SAT, UNSAT, ADR/MCC, and assignment-satisfaction metrics
    on (model, N). Output columns:

      model, N,
      precision_sat, recall_sat, f1_sat, accuracy,
      precision_unsat, recall_unsat, f1_unsat,
      MCC, ADR,
      Assignments_Satisfied_Rate

    Parameters
    ----------
    df_SAT : pd.DataFrame
        Must contain columns ['model','N','Precision','Recall','F1-score']
        and optionally 'Accuracy'.
    df_UNSAT : pd.DataFrame
        Must contain columns ['model','N','Precision','Recall','F1-score'].
    ADR_DF : pd.DataFrame
        Must contain columns ['model','N','MCC','ADR'].
    ASSIGNMENT_DF : pd.DataFrame
        Must contain ['model','N', assignment_col].
    assignment_col : str
        Column name in ASSIGNMENT_DF representing Assignments_Satisfied_Rate.
    export_path : str or None
        If provided, save the merged table to this Excel path.

    Returns
    -------
    pd.DataFrame
        Merged wide table aligned by (model, N).
    """
    def _safe_pick(df, cols):
        have = [c for c in cols if c in df.columns]
        return df[have].copy() if have else pd.DataFrame(columns=cols)

    # SAT view
    sat_cols_src = ["model", "N", "Precision", "Recall", "F1-score", "Accuracy"]
    sat_df = _safe_pick(df_SAT, sat_cols_src)
    sat_df = sat_df.rename(columns={
        "Precision": "precision_sat",
        "Recall": "recall_sat",
        "F1-score": "f1_sat",
        "Accuracy": "accuracy"
    })

    # UNSAT view
    unsat_cols_src = ["model", "N", "Precision", "Recall", "F1-score"]
    unsat_df = _safe_pick(df_UNSAT, unsat_cols_src)
    unsat_df = unsat_df.rename(columns={
        "Precision": "precision_unsat",
        "Recall": "recall_unsat",
        "F1-score": "f1_unsat"
    })

    # ADR/MCC view
    adr_cols_src = ["model", "N", "MCC", "ADR"]
    adr_df = _safe_pick(ADR_DF, adr_cols_src)

    # Assignment satisfied rate view
    assign_cols_src = ["model", "N", assignment_col]
    assign_df = _safe_pick(ASSIGNMENT_DF, assign_cols_src)
    assign_df = assign_df.rename(columns={assignment_col: "Assignments_Satisfied_Rate"})

    # Outer merge on (model, N)
    merged = sat_df.merge(unsat_df, on=["model", "N"], how="outer")
    merged = merged.merge(adr_df, on=["model", "N"], how="outer")
    merged = merged.merge(assign_df, on=["model", "N"], how="outer")

    # Sort for readability
    if "model" in merged.columns and "N" in merged.columns:
        merged = merged.sort_values(by=["model", "N"]).reset_index(drop=True)

    # Optional export
    if export_path:
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        except Exception:
            pass
        merged.to_excel(export_path, index=False)
        print("Merged metrics Excel:", export_path)

    return merged

export_path = save_base + "_merged_metrics.xlsx"
build_merged_metrics(df_SAT, df_UNSAT, ADR_DF, ASSIGNMENT_DF, ASSIGNMENT_col, export_path=export_path)











# ========= Additional figure: draw only the first 3x3 panels =========

fig_3x3, axes_3x3 = plt.subplots(3, 3, figsize=(20, 14))
axes_3x3 = axes_3x3.ravel()

letters_3x3 = list("abcdefghi")

# Keep only the first 9 panels
panels_3x3 = [
    # Row 1: SAT
    ("Precision SAT", "Precision", df_SAT),
    ("Recall SAT", "Recall", df_SAT),
    ("F1 SAT", "F1-score", df_SAT),

    # Row 2: UNSAT
    ("Precision UNSAT", "Precision", df_UNSAT),
    ("Recall UNSAT", "Recall", df_UNSAT),
    ("F1 UNSAT", "F1-score", df_UNSAT),

    # Row 3: Accuracy, MCC, ADR
    ("Accuracy", "Accuracy", df_SAT),
    ("MCC", "MCC", ADR_DF),
    ("ADR", "ADR", ADR_DF),
]

for idx, (title, col, dframe) in enumerate(panels_3x3):
    ax = axes_3x3[idx]

    for m in MODELS:
        st = style_for(m)
        if st is None:
            continue
        _, color, ls, marker = st

        sub = dframe[dframe["model_clean"].str.lower() == m.lower()].copy()
        if sub.empty:
            continue
        sub = sub[sub["N"].isin(x_order)].copy()
        if sub.empty:
            continue

        sub["xpos"] = sub["N"].map(xpos_map)
        sub = sub.sort_values("xpos")

        xvals = sub["xpos"].astype(float).values
        yvals = sub[col].astype(float).values

        # Keep the same jitter logic for the Recall SAT panel
        if idx == 1:
            xjit = JITTER_RECALL_SAT_X.get(m.lower(), 0.0)
            if xjit != 0.0:
                xvals = np.array([
                    xv + (xjit if x_order[int(xv)] in JITTER_N_ONLY else 0.0)
                    for xv in xvals
                ])
            yjit = YJITTER_RECALL_SAT.get(m.lower(), 0.0)
            if yjit != 0.0:
                xpos_list = sub["xpos"].astype(int).values
                yvals = np.array([
                    y + (yjit if x_order[xi] in JITTER_N_ONLY else 0.0)
                    for y, xi in zip(yvals, xpos_list)
                ])
                yvals = np.clip(yvals, 0.0, 1.0)

        ax.plot(
            xvals, yvals,
            label=legend_label(m),
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.4, clip_on=False
        )

    ax.set_title(f"({letters_3x3[idx]}) {title}")
    ax.set_xlabel("N")
    ax.set_ylabel(title.split()[0])

    ax.set_xticks(xpos_ticks)
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.3, len(x_order) - 0.7)

    if title == "MCC":
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#999999", lw=1.2)
    else:
        ax.set_ylim(-0.05, 1.05)

    ax.grid(True, linewidth=1)

# Rebuild legend for the 3x3 figure
handles_3x3, labels_3x3 = [], []
for m in MODELS:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)
    h = Line2D(
        [0], [0],
        color=color, linestyle=ls, marker=marker,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.4, label=lab
    )
    handles_3x3.append(h)
    labels_3x3.append(lab)

# Move the legend upward compared with the original lower position
fig_3x3.legend(
    handles=handles_3x3,
    labels=labels_3x3,
    loc="lower center",
    ncol=4,
    frameon=True,
    columnspacing=1.4,
    handletextpad=0.6,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig_3x3.subplots_adjust(hspace=0.35, wspace=0.28)

# Save the new 3x3 figure
save_base_3x3 = save_base + "_3x3_only"

pdf_path_3x3 = save_base_3x3 + ".pdf"
svg_path_3x3 = save_base_3x3 + ".svg"
png_path_3x3 = save_base_3x3 + ".png"

fig_3x3.savefig(pdf_path_3x3, dpi=600, bbox_inches="tight")
fig_3x3.savefig(svg_path_3x3, dpi=600, bbox_inches="tight")
fig_3x3.savefig(png_path_3x3, dpi=600, bbox_inches="tight")

print("Saved PDF (3x3 only):", pdf_path_3x3)
print("Saved SVG (3x3 only):", svg_path_3x3)
print("Saved PNG (3x3 only):", png_path_3x3)