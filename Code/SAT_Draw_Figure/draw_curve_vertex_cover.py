import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ========= Path configuration =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)   # Go up from SAT_Draw_Figure to Code
BASE_DIR = os.path.join(CODE_DIR, "Analysis_Result_Collection")

xls_path = os.path.join(BASE_DIR, "SAT_figures.xlsx")
save_dir = os.path.join(BASE_DIR, "Figure_in_paper", "Vertex_Cover")
save_base = os.path.join(save_dir, "Vertex_Cover")

print("Script directory:", SCRIPT_DIR)
print("Code root directory:", CODE_DIR)
print("Input Excel path:", xls_path)
print("Output directory:", save_dir)
print("Output base path:", save_base)

# ========= Read equivalence-rate data (two columns, unchanged) =========
equivalence_rate_df = pd.read_excel(
    xls_path,
    sheet_name="Match_Vertex_Cover_And_CNF_Pred",
    skiprows=0
)
equivalence_rate_df = equivalence_rate_df.rename(columns={
    "cnf_and_VC_llm_answer_yes_have_same_prediction_true_ratio": "vc_cnf_equivalence",
    "cnf_and_vc_and_label_are_same_true_ratio": "vc_cnf_label_equivalence"
})
equivalence_col = "vc_cnf_equivalence"
equivalence_col_2 = "vc_cnf_label_equivalence"
need_cols_eq = ["model", "N", equivalence_col, equivalence_col_2]
equivalence_rate_df = equivalence_rate_df.dropna(subset=need_cols_eq)[need_cols_eq].copy()
equivalence_rate_df["N"] = pd.to_numeric(equivalence_rate_df["N"], errors="coerce")
equivalence_rate_df[equivalence_col] = pd.to_numeric(equivalence_rate_df[equivalence_col], errors="coerce")
equivalence_rate_df[equivalence_col_2] = pd.to_numeric(equivalence_rate_df[equivalence_col_2], errors="coerce")
equivalence_rate_df = equivalence_rate_df.dropna(subset=["N"])

# ========= Read unified data source Vertex_Cover_ADR =========
# Includes precision/recall/f1/accuracy/ADR/MCC
df_vc_adr = pd.read_excel(xls_path, sheet_name="Vertex_Cover_ADR")
df_vc_adr = df_vc_adr.rename(columns={
    "model_select": "model",
    "ADR (Accurate Differentiation Rate)": "ADR",
    "ACC": "accuracy",
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "F1-score": "f1",
})
cols_wanted = ["model", "N", "accuracy", "precision", "recall", "f1", "ADR", "MCC"]
exist_cols = [c for c in cols_wanted if c in df_vc_adr.columns]
df_vc_adr = df_vc_adr.dropna(subset=["model", "N"])[exist_cols].copy()
df_vc_adr["N"] = pd.to_numeric(df_vc_adr["N"], errors="coerce")
for c in exist_cols:
    if c not in ("model", "N"):
        df_vc_adr[c] = pd.to_numeric(df_vc_adr[c], errors="coerce")
df_vc_adr = df_vc_adr.dropna(subset=["N"])

# ========= Read Vertex_Cover_Assignments =========
# Used for the 9th subplot
df_assign = pd.read_excel(xls_path, sheet_name="Vertex_Cover_Assignments")
df_assign = df_assign.rename(columns={
    "model_select": "model"  # Safe fallback if the original column is already "model"
})
assign_col = "yes_cover_valid_rate"
df_assign = df_assign.dropna(subset=["model", "N", assign_col])[["model", "N", assign_col]].copy()
df_assign["N"] = pd.to_numeric(df_assign["N"], errors="coerce")
df_assign[assign_col] = pd.to_numeric(df_assign[assign_col], errors="coerce")
df_assign = df_assign.dropna(subset=["N"])

# ========= Style and filtering helpers =========
DATE_SUFFIX_RE = re.compile(r'[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}$')

def legend_label(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    if low.startswith("claude"):
        s = DATE_SUFFIX_RE.sub("", s)
        s = re.sub(r"[-_]?20\d{6,8}$", "", s)
    if "deepseek" in low:
        s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s

COLOR_CLAUDE = "#7b3fc8"
COLOR_DEEPSEEK = "#2ca02c"
COLOR_OAI_RED = "#d62728"
COLOR_OAI_YELL = "#ffb000"

LS_CLAUDE = "--"
LS_DEEPSEEK = (0, (10, 4))
LS_OPENAI = "-"

DEEPSEEK_SET = {"deepseek-chat": "*", "deepseek-reasoner": "s"}
OPENAI_RED_SET = {"gpt-5": "o", "o1": "D", "o3-mini": "X"}
OPENAI_YELLOW_SET = {"gpt-4.1": "P", "gpt-3.5-turbo-0125": "d", "gpt-4o-latest": "h"}

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

def _in_whitelist(ml: str) -> bool:
    ml = ml.lower()
    if re.search(r"\bgpt-4-turbo\b", ml):
        return False
    if ml == "gpt-4o":
        return False
    if re.search(r"\bchatgpt-4o-latest\b", ml):
        return False
    if ml.startswith("claude"):
        return True
    if "deepseek" in ml:
        return True
    if ml in OPENAI_RED_SET:
        return True
    if any(p in ml for p in ["gpt-4.1", "gpt-3.5-turbo-0125", "gpt-4o-latest"]):
        return True
    return False

def style_for(model: str):
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

def _clean_and_filter(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out["model_clean"] = out[model_col].astype(str).str.strip()
    low = out["model_clean"].str.lower()
    mask = ~low.str.contains(r"\bgpt-4-turbo\b", na=False)
    mask &= (low != "gpt-4o")
    mask &= ~low.str.contains(r"\bchatgpt-4o-latest\b", na=False)
    out = out[mask]
    out["model_clean"] = out["model_clean"].apply(legend_label)
    out = out[out["model_clean"].str.lower().map(_in_whitelist)]
    if "N" in out.columns:
        out.sort_values(["model_clean", "N"], inplace=True)
        out.drop_duplicates(subset=["model_clean", "N"], keep="last", inplace=True)
    return out

equivalence_rate_df = _clean_and_filter(equivalence_rate_df, "model")
df_vc_adr = _clean_and_filter(df_vc_adr, "model")
df_assign = _clean_and_filter(df_assign, "model")

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

# Include assignment models in the global legend and plotting model set
models_all = pd.Series(pd.concat([
    equivalence_rate_df["model_clean"],
    df_vc_adr["model_clean"],
    df_assign["model_clean"]
], ignore_index=True))
MODELS = ordered_models(models_all)

# ========= Plot style =========
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 28,
    "axes.titlesize": 24,
    "axes.labelsize": 28,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 24,
    "lines.linewidth": 3.0,
    "lines.markersize": 9,
})

# ========= 3×3 canvas =========
# The 9th subplot draws yes_cover_valid_rate
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

# Order:
# Row 1: Precision, Recall, F1 (from Vertex_Cover_ADR)
# Row 2: Accuracy (from Vertex_Cover_ADR), vc_cnf_equivalence, vc_cnf_label_equivalence
# Row 3: ADR, MCC, yes_cover_valid_rate
panels = [
    ("Precision", "precision", df_vc_adr),
    ("Recall", "recall", df_vc_adr),
    ("F1", "f1", df_vc_adr),
    ("Accuracy", "accuracy", df_vc_adr),
    ("vc_cnf_equivalence", "vc_cnf_equivalence", equivalence_rate_df),
    ("vc_cnf_label_equivalence", "vc_cnf_label_equivalence", equivalence_rate_df),
    ("ADR", "ADR", df_vc_adr),
    ("MCC", "MCC", df_vc_adr),
    ("yes_cover_valid_rate", "yes_cover_valid_rate", df_assign),
]

# Evenly spaced x-axis
x_order = [5, 8, 10, 25]
xpos_map = {v: i for i, v in enumerate(x_order)}
xpos_ticks = list(range(len(x_order)))
letters = list("abcdefghi")

# ========= Draw 9 subplots =========
for idx, (title, col, dframe) in enumerate(panels, start=0):
    ax = axes[idx]
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

        if col not in sub.columns:
            continue

        ax.plot(
            sub["xpos"],
            sub[col],
            label=legend_label(m),
            color=color,
            linestyle=ls,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4
        )

    ax.set_title(f"({letters[idx]}) {title}")
    ax.set_xlabel("N")
    ax.set_ylabel(title)
    ax.set_xticks(xpos_ticks)
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.2, len(x_order) - 0.8)

    # Metrics in [0, 1]
    if col in {
        "accuracy", "precision", "recall", "f1",
        "vc_cnf_equivalence", "vc_cnf_label_equivalence",
        "ADR", "yes_cover_valid_rate"
    }:
        ax.set_ylim(-0.05, 1.05)

    # MCC in [-1, 1]
    if col == "MCC":
        ax.set_ylim(-1.05, 1.05)

    ax.grid(True, linewidth=1)

# ========= Bottom legend for the whole figure =========
handles, labels = [], []
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
    handles.append(h)
    labels.append(lab)

fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    ncol=3,
    frameon=True,
    columnspacing=1.4,
    handletextpad=0.6,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.subplots_adjust(hspace=0.34, wspace=0.28)

# ========= Save figure =========
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

def build_merged_metrics_vertex_cover(
    df_vc_adr: pd.DataFrame,
    equivalence_rate_df: pd.DataFrame,
    df_assign: pd.DataFrame,
    filter_models: list | None = None,
    filter_N: list | None = None,
    export_path: str | None = None
) -> pd.DataFrame:
    """
    Merge the three Vertex Cover data sources horizontally using (model, N) as keys:

      - df_vc_adr:
        precision / recall / f1 / accuracy / ADR / MCC
      - equivalence_rate_df:
        vc_cnf_equivalence / vc_cnf_label_equivalence
      - df_assign:
        yes_cover_valid_rate

    Output columns (included if present in the source tables):
      model, N,
      precision, recall, f1, accuracy, ADR, MCC,
      vc_cnf_equivalence, vc_cnf_label_equivalence,
      yes_cover_valid_rate
    """
    import os
    import pandas as pd
    import numpy as np

    def _model_key(df: pd.DataFrame) -> str:
        if "model_clean" in df.columns:
            return "model_clean"
        if "model" in df.columns:
            return "model"
        raise ValueError("Input DataFrame must contain either 'model' or 'model_clean'.")

    key_vc = _model_key(df_vc_adr)
    key_eq = _model_key(equivalence_rate_df)
    key_asg = _model_key(df_assign)

    def _safe_pick(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        have = [c for c in cols if c in df.columns]
        if not have:
            needed_keys = [c for c in cols if c in (key_vc, key_eq, key_asg, "N")]
            return pd.DataFrame(columns=needed_keys)

        out = df[have].copy()

        if "N" in out.columns:
            out["N"] = pd.to_numeric(out["N"], errors="coerce")

        for c in have:
            if c not in (key_vc, key_eq, key_asg, "N"):
                out[c] = pd.to_numeric(out[c], errors="coerce")

        if "N" in out.columns:
            out = out.dropna(subset=["N"])

        return out

    vc_cols = [key_vc, "N", "precision", "recall", "f1", "accuracy", "ADR", "MCC"]
    vc = _safe_pick(df_vc_adr, vc_cols)

    eq_cols = [key_eq, "N", "vc_cnf_equivalence", "vc_cnf_label_equivalence"]
    eq = _safe_pick(equivalence_rate_df, eq_cols)

    asg_cols = [key_asg, "N", "yes_cover_valid_rate"]
    asg = _safe_pick(df_assign, asg_cols)

    if key_eq != key_vc and key_eq in eq.columns:
        eq = eq.rename(columns={key_eq: key_vc})
    if key_asg != key_vc and key_asg in asg.columns:
        asg = asg.rename(columns={key_asg: key_vc})

    merged = vc.merge(eq, on=[key_vc, "N"], how="outer") \
               .merge(asg, on=[key_vc, "N"], how="outer")

    merged = merged.rename(columns={key_vc: "model"})

    if filter_models is not None:
        fm_low = {str(m).lower() for m in filter_models}
        merged = merged[merged["model"].astype(str).str.lower().isin(fm_low)]

    if filter_N is not None:
        merged = merged[merged["N"].isin(filter_N)]

    merged = merged.sort_values(["model", "N"]).reset_index(drop=True)

    col_order = [
        "model", "N",
        "precision", "recall", "f1", "accuracy", "ADR", "MCC",
        "vc_cnf_equivalence", "vc_cnf_label_equivalence",
        "yes_cover_valid_rate"
    ]
    final_cols = [c for c in col_order if c in merged.columns] + \
                 [c for c in merged.columns if c not in col_order]
    merged = merged[final_cols]

    if export_path:
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        except Exception:
            pass
        merged.to_excel(export_path, index=False)
        print("Merged metrics Excel:", export_path)

    return merged

# ========= Example usage =========
merged_export_path = save_base + "_merged_metrics.xlsx"

merged_vc = build_merged_metrics_vertex_cover(
    df_vc_adr=df_vc_adr,
    equivalence_rate_df=equivalence_rate_df,
    df_assign=df_assign,
    filter_models=MODELS,
    filter_N=[5, 8, 10, 25],
    export_path=merged_export_path
)

print("Merged VC metrics shape:", merged_vc.shape)
print("Merged VC metrics saved to:", merged_export_path)













# ========= Additional figure: remove yes_cover_valid_rate and place legend in the last panel =========

# Fixed legend order requested by the user
LEGEND_ORDER_FOR_LAST_PANEL = [
    "deepseek-reasoner",
    "gpt-5",
    "o3-mini",
    "o1",
    "gpt-3.5-turbo-0125",
    "gpt-4.1",
    "gpt-4o-latest",
]

# Keep only models that actually exist in the current plotting data
MODELS_LAST_PANEL = [
    m for m in LEGEND_ORDER_FOR_LAST_PANEL
    if any(str(x).lower() == m.lower() for x in MODELS)
]

# Create a new 3x3 figure
fig_last_legend, axes_last_legend = plt.subplots(3, 3, figsize=(20, 15))
axes_last_legend = axes_last_legend.ravel()

# Use only the first 8 panels; the 9th panel is reserved for the legend
panels_last_legend = [
    ("Precision", "precision", df_vc_adr),
    ("Recall", "recall", df_vc_adr),
    ("F1", "f1", df_vc_adr),
    ("Accuracy", "accuracy", df_vc_adr),
    ("vc_cnf_equivalence", "vc_cnf_equivalence", equivalence_rate_df),
    ("vc_cnf_label_equivalence", "vc_cnf_label_equivalence", equivalence_rate_df),
    ("ADR", "ADR", df_vc_adr),
    ("MCC", "MCC", df_vc_adr),
]

letters_8 = list("abcdefgh")

# Draw the first 8 panels
for idx, (title, col, dframe) in enumerate(panels_last_legend, start=0):
    ax = axes_last_legend[idx]

    for m in MODELS_LAST_PANEL:
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

        if col not in sub.columns:
            continue

        ax.plot(
            sub["xpos"],
            sub[col],
            label=legend_label(m),
            color=color,
            linestyle=ls,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4
        )

    ax.set_title(f"({letters_8[idx]}) {title}")
    ax.set_xlabel("N")
    ax.set_ylabel(title)
    ax.set_xticks(xpos_ticks)
    ax.set_xticklabels(x_order)
    ax.set_xlim(-0.2, len(x_order) - 0.8)

    if col in {
        "accuracy", "precision", "recall", "f1",
        "vc_cnf_equivalence", "vc_cnf_label_equivalence",
        "ADR"
    }:
        ax.set_ylim(-0.05, 1.05)

    if col == "MCC":
        ax.set_ylim(-1.05, 1.05)

    ax.grid(True, linewidth=1)

# Use the last panel for the legend
ax_legend = axes_last_legend[8]
ax_legend.axis("off")

legend_handles = []
legend_labels = []

for m in MODELS_LAST_PANEL:
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
    legend_handles.append(h)
    legend_labels.append(lab)

# Place legend inside the last subplot, vertically
ax_legend.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="center",
    ncol=1,
    frameon=True,
    columnspacing=1.0,
    handletextpad=0.6,
    borderpad=0.8
)

plt.tight_layout()
fig_last_legend.subplots_adjust(hspace=0.34, wspace=0.28)

# Save the new figure with a new filename
save_base_last_legend = save_base + "_legend_in_last_panel"

pdf_path_last_legend = save_base_last_legend + ".pdf"
svg_path_last_legend = save_base_last_legend + ".svg"
png_path_last_legend = save_base_last_legend + ".png"

fig_last_legend.savefig(pdf_path_last_legend, dpi=600, bbox_inches="tight")
fig_last_legend.savefig(svg_path_last_legend, dpi=600, bbox_inches="tight")
fig_last_legend.savefig(png_path_last_legend, dpi=600, bbox_inches="tight")

print("Saved PDF (legend in last panel):", pdf_path_last_legend)
print("Saved SVG (legend in last panel):", svg_path_last_legend)
print("Saved PNG (legend in last panel):", png_path_last_legend)