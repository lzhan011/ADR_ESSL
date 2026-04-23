import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef
from collections import Counter
import numpy as np
import os


def set_UNSAT_as_positive(y_true, y_pred):
    y_true_UNSAT_as_Positive = [not x for x in y_true]
    y_pred_UNSAT_as_Positive = [not x for x in y_pred]
    precision_UNSAT_as_Positive = precision_score(y_true_UNSAT_as_Positive, y_pred_UNSAT_as_Positive)
    recall_UNSAT_as_Positive = recall_score(y_true_UNSAT_as_Positive, y_pred_UNSAT_as_Positive)
    f1_UNSAT_as_Positive = f1_score(y_true_UNSAT_as_Positive, y_pred_UNSAT_as_Positive)
    mcc_UNSAT_as_Positive = matthews_corrcoef(y_true_UNSAT_as_Positive, y_pred_UNSAT_as_Positive)
    return precision_UNSAT_as_Positive, recall_UNSAT_as_Positive, f1_UNSAT_as_Positive, mcc_UNSAT_as_Positive


def get_one_model_one_version_result(Predictions_before, Predictions_after):
    y_labels_before = [0] * len(Predictions_before)
    y_labels_after = [1] * len(Predictions_after)  # 1 means SAT; after fix, the CNF becomes SAT
    one_row = {}
    y_true = y_labels_before + y_labels_after
    y_pred = Predictions_before + Predictions_after
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision_UNSAT_as_Positive, recall_UNSAT_as_Positive, f1_UNSAT_as_Positive, mcc_UNSAT_as_Positive = set_UNSAT_as_positive(y_true, y_pred)
    before_correct = 0
    after_correct = 0
    before_wrong = 0
    after_wrong = 0

    same_prediction = 0
    different_prediction = 0
    same_prediction_positive = 0
    same_prediction_negative = 0

    different_prediction_correct = 0
    different_prediction_incorrect = 0
    for i in range(len(Predictions_before)):
        Predictions_before_i = Predictions_before[i]
        y_labels_before_i = y_labels_before[i]

        Predictions_after_i = Predictions_after[i]
        y_labels_after_i = y_labels_after[i]
        if Predictions_before_i == y_labels_before_i:
            before_correct += 1
        else:
            before_wrong += 1

        if Predictions_after_i == y_labels_after_i:
            after_correct += 1
        else:
            after_wrong += 1

        if Predictions_before_i == Predictions_after_i:
            same_prediction += 1

            if Predictions_before_i:
                same_prediction_positive += 1
            else:
                same_prediction_negative += 1

        else:
            different_prediction += 1

            if Predictions_before_i == y_labels_before_i and Predictions_after_i == y_labels_after_i:
                different_prediction_correct += 1
            else:
                different_prediction_incorrect += 1

    DR = different_prediction / len(Predictions_before)
    ADR = different_prediction_correct / len(Predictions_before)
    SDR = 0.5 * DR + 0.5 * ADR
    CR = same_prediction / len(Predictions_before)

    DR = round(DR, 2)
    ADR = round(ADR, 2)
    SDR = round(SDR, 2)
    CR = round(CR, 2)

    one_row['C (Number of Confused)'] = same_prediction
    one_row['CP (Number of Confused-positive)'] = same_prediction_positive
    one_row['CN (Number of Confused-negative)'] = same_prediction_negative
    one_row['S(Number of Separated)'] = different_prediction
    one_row['SC (Number of Separated-correct)'] = different_prediction_correct
    one_row['SI (Number of Separated-incorrect)'] = different_prediction_incorrect
    one_row['DR (Differentiation Rate)'] = DR
    one_row['ADR (Accurate Differentiation Rate)'] = ADR
    one_row['SDR (Symmetric Differentiation Rate)'] = SDR
    one_row['CR (Confusion Rate)'] = CR
    one_row['MCC'] = mcc
    one_row['accuracy'] = accuracy
    one_row['precision'] = precision
    one_row['recall'] = recall
    one_row['f1'] = f1
    one_row["precision_UNSAT_as_Positive"] = precision_UNSAT_as_Positive
    one_row["recall_UNSAT_as_Positive"] = recall_UNSAT_as_Positive
    one_row["f1_UNSAT_as_Positive"] = f1_UNSAT_as_Positive
    one_row["mcc_UNSAT_as_Positive"] = mcc_UNSAT_as_Positive
    return one_row


def get_pairs(all_files) -> dict:
    """
    Return a dictionary in the following form:
      {
        'cnf_k3_N8_L28_alpha3.5_inst1065.txt': {
            'orig':  'cnf_k3_N8_L28_alpha3.5_inst1065.txt',
            'fixed': 'cnf_k3_N8_L28_alpha3.5_inst1065_RC2_fixed.cnf'
        },
        ...
      }
    """
    pairs = {}
    for f in sorted(all_files):
        base, ext = os.path.splitext(f)

        if base.endswith("_RC2_fixed"):
            key = base[:-len("_RC2_fixed")] + ".txt"  # Normalize the key extension to .txt
            d = pairs.setdefault(key, {})
            d["fixed"] = f
        else:
            if ext.lower() == ".txt":
                key = f
                d = pairs.setdefault(key, {})
                d["orig"] = f
    return pairs


def convert_SAT_True_False(x):
    return True if x == "SAT" else False


def get_our_metrics(df):
    """
    Use only PerSample_all_model_all_N to compute and return
    our_metrics_all_model_all_N.

    Output columns include:
    ['model', 'N', 'accuracy', 'MCC', 'ADR (Accurate Differentiation Rate)',
     'precision', 'recall', 'f1',
     'precision_UNSAT_as_Positive', 'recall_UNSAT_as_Positive',
     'f1_UNSAT_as_Positive', ...]
    """
    df = df.copy()
    df['pred_llm_yes_boolean'] = df['pred_llm_yes'].apply(convert_SAT_True_False)
    model_unique = df['model'].unique()

    rows = []
    for model in model_unique:
        one_model_res = df[df['model'] == model]
        N_unique = one_model_res['N_meta'].unique()
        for N in N_unique:
            sub = one_model_res[one_model_res['N_meta'] == N]
            filenames = sub['filename'].tolist()
            pairs = get_pairs(filenames)

            Predictions_before, Predictions_after = [], []
            for k, v in pairs.items():
                if "orig" in v and "fixed" in v:
                    f_orig = v["orig"]
                    f_fixed = v["fixed"]
                    pred_orig = sub[sub['filename'] == f_orig]['pred_llm_yes_boolean'].iloc[0]
                    pred_fixed = sub[sub['filename'] == f_fixed]['pred_llm_yes_boolean'].iloc[0]
                    Predictions_before.append(pred_orig)
                    Predictions_after.append(pred_fixed)

            if len(Predictions_before) > 0:
                one_row = get_one_model_one_version_result(Predictions_before, Predictions_after)
                one_row['model'] = model
                one_row['N'] = N
                rows.append(one_row)

    res = pd.DataFrame(rows)
    if not res.empty:
        cols = res.columns.tolist()
        new_order = ['model', 'N'] + [c for c in cols if c not in ('model', 'N')]
        res = res[new_order]
    return res


# ========= Plot styling utilities (unchanged) =========
DATE_SUFFIX_RE = re.compile(r'[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}$')  # -2025-08-30 / _20250830

def legend_label(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    if low.startswith("claude"):
        s = DATE_SUFFIX_RE.sub("", s)
        s = re.sub(r"[-_]?20\d{6,8}$", "", s)
    if "deepseek" in low:
        s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s

COLOR_CLAUDE = "#7b3fc8"   # purple
COLOR_DEEPSEEK = "#2ca02c"   # green
COLOR_OAI_RED = "#d62728"   # red
COLOR_OAI_YELL = "#ffb000"   # yellow

LS_CLAUDE = "--"
LS_DEEPSEEK = (0, (10, 4))   # long dashed line
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
    if ml.startswith("claude"):
        return True
    if "deepseek" in ml:
        return True
    if ml in OPENAI_RED_SET:
        return True
    if any(x in ml for x in ["gpt-4.1", "gpt-3.5-turbo-0125", "gpt-4o-latest"]):
        return True
    return False

def style_for(model: str):
    """Return (group, color, linestyle, marker); return None if not in the whitelist."""
    m = str(model).strip()
    ml = m.lower()
    if re.search(r"\bgpt-4-turbo\b", ml):
        return None
    if ml == "gpt-4o":
        return None
    if re.search(r"\bchatgpt-4o-latest\b", ml):
        return None
    if not _in_whitelist(ml):
        return None

    if ml.startswith("claude"):
        return ("claude", COLOR_CLAUDE, LS_CLAUDE, _claude_marker(ml))
    if ml in DEEPSEEK_SET or "deepseek" in ml:
        return ("deepseek", COLOR_DEEPSEEK, LS_DEEPSEEK, DEEPSEEK_SET.get(ml, "*"))
    if ml in OPENAI_RED_SET:
        return ("openai-red", COLOR_OAI_RED, LS_OPENAI, OPENAI_RED_SET[ml])
    return ("openai-yellow", COLOR_OAI_YELL, LS_OPENAI, OPENAI_YELLOW_SET.get(ml, _openai_yellow_marker(ml)))

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


# ========= Path configuration =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)   # Go up from SAT_Draw_Figure to Code root
BASE_DIR = os.path.join(CODE_DIR, "Analysis_Result_Collection")

# INPUT_FILE = os.path.join(BASE_DIR, "summary_metrics.xlsx")
SAVE_DIR = os.path.join(BASE_DIR, "Figure_in_paper", "3d_packing")
INPUT_FILE = os.path.join(BASE_DIR, "Figure_in_paper", "3d_packing", "summary_metrics.xlsx")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_BASE = os.path.join(SAVE_DIR, "packing_pred_llm_yes_3x3")
SAVE_BASE_4x3 = os.path.join(SAVE_DIR, "packing_pred_llm_yes_3x3_4x3")

print("Script directory:", SCRIPT_DIR)
print("Code root directory:", CODE_DIR)
print("Input Excel path:", INPUT_FILE)
print("Output directory:", SAVE_DIR)
print("Output base path (3x3):", SAVE_BASE)
print("Output base path (4x3):", SAVE_BASE_4x3)

# ========= Read only one sheet =========
PerSample_all_model_all_N = pd.read_excel(INPUT_FILE, sheet_name='PerSample_all_model_all_N')

# ========= Compute all metrics from PerSample =========
our_metrics_all_model_all_N = get_our_metrics(PerSample_all_model_all_N)

# Build the five data tables used for plotting
# Accuracy
pred_llm_yes_ACC = our_metrics_all_model_all_N[['model', 'N', 'accuracy']].copy()

# SAT view (positive class = SAT)
pred_SAT = our_metrics_all_model_all_N[['model', 'N', 'precision', 'recall', 'f1']].copy()

# UNSAT view (positive class = UNSAT)
pred_UNSAT = our_metrics_all_model_all_N[
    ['model', 'N', 'precision_UNSAT_as_Positive', 'recall_UNSAT_as_Positive', 'f1_UNSAT_as_Positive']
].rename(columns={
    'precision_UNSAT_as_Positive': 'precision',
    'recall_UNSAT_as_Positive': 'recall',
    'f1_UNSAT_as_Positive': 'f1'
})

# ADR and MCC
pred_ADR = our_metrics_all_model_all_N[['model', 'N', 'ADR (Accurate Differentiation Rate)']].rename(
    columns={'ADR (Accurate Differentiation Rate)': 'ADR'}
)
pred_MCC = our_metrics_all_model_all_N[['model', 'N', 'MCC']].copy()

# ========= Model order and x-axis =========
MODELS = ordered_models(pd.concat([
    pred_llm_yes_ACC['model'], pred_SAT['model'], pred_UNSAT['model'], pred_ADR['model'], pred_MCC['model']
], ignore_index=True))

X_ORDER = sorted(set(pd.concat([
    pred_llm_yes_ACC['N'], pred_SAT['N'], pred_UNSAT['N'], pred_ADR['N'], pred_MCC['N']
], ignore_index=True).dropna().unique()))
xpos_map = {v: i for i, v in enumerate(X_ORDER)}
xpos_ticks = list(range(len(X_ORDER)))

# ========= Global plot style (unchanged) =========
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 26, "axes.titlesize": 26, "axes.labelsize": 26,
    "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 22,
    "lines.linewidth": 3.0, "lines.markersize": 9,
})

fig, axes = plt.subplots(3, 3, figsize=(22, 16))
axes = axes.ravel()

# ========= 3×3 panel order (unchanged) =========
panels = [
    # Row 1: SAT
    ("(a) Precision (SAT)", pred_SAT, "precision"),
    ("(b) Recall (SAT)", pred_SAT, "recall"),
    ("(c) F1 (SAT)", pred_SAT, "f1"),

    # Row 2: UNSAT
    ("(d) Precision (UNSAT)", pred_UNSAT, "precision"),
    ("(e) Recall (UNSAT)", pred_UNSAT, "recall"),
    ("(f) F1 (UNSAT)", pred_UNSAT, "f1"),

    # Row 3: Accuracy, ADR, MCC
    ("(g) Accuracy", pred_llm_yes_ACC, "accuracy"),
    ("(h) ADR", pred_ADR, "ADR"),
    ("(i) MCC", pred_MCC, "MCC"),
]

def _panel_ylim(value_col: str):
    if value_col.upper() == "MCC":
        return (-1.05, 1.05)
    return (0, 1.05)

def _plot_panel(ax, title, df_src, value_col: str):
    ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_ylabel(value_col.upper() if value_col else "")
    ax.set_xticks(xpos_ticks)
    ax.set_xticklabels(X_ORDER)
    ax.set_xlim(-0.2, len(X_ORDER) - 0.8)
    ymin, ymax = _panel_ylim(value_col)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linewidth=1.1)

    if df_src is None or df_src.empty:
        return

    for m in MODELS:
        st = style_for(m)
        if st is None:
            continue
        _, color, ls, marker = st
        sub = df_src[df_src['model'].astype(str).str.lower() == m.lower()].copy()
        if sub.empty:
            continue
        sub = sub[sub['N'].isin(X_ORDER)]
        if sub.empty:
            continue
        sub['xpos'] = sub['N'].map(xpos_map)
        sub = sub.sort_values('xpos')
        y = pd.to_numeric(sub[value_col], errors='coerce')
        ax.plot(
            sub['xpos'], y,
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.6,
            label=legend_label(m)
        )

for ax, (title, df_src, col) in zip(axes, panels):
    _plot_panel(ax, title, df_src, col)

# ========= Bottom legend (unchanged) =========
handles, labels, seen = [], [], set()
for m in MODELS:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)
    if lab in seen:
        continue
    seen.add(lab)
    h = Line2D([0], [0], color=color, linestyle=ls, marker=marker,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.6, label=lab)
    handles.append(h)
    labels.append(lab)

fig.legend(handles=handles, labels=labels,
           loc="lower center", ncol=4, frameon=True,
           columnspacing=1.4, handletextpad=0.6)

plt.tight_layout(rect=[0, 0.10, 1, 1])  # Reserve space for the bottom legend
fig.subplots_adjust(hspace=0.34, wspace=0.28)

# ========= Export =========
os.makedirs(SAVE_DIR, exist_ok=True)

pdf_path_3x3 = SAVE_BASE + ".pdf"
svg_path_3x3 = SAVE_BASE + ".svg"
png_path_3x3 = SAVE_BASE + ".png"

fig.savefig(pdf_path_3x3, dpi=600, bbox_inches="tight")
fig.savefig(svg_path_3x3, dpi=600, bbox_inches="tight")
fig.savefig(png_path_3x3, dpi=600, bbox_inches="tight")

print("Saved PDF (3x3):", pdf_path_3x3)
print("Saved SVG (3x3):", svg_path_3x3)
print("Saved PNG (3x3):", png_path_3x3)


# ==================== Additional part: compute P(assignment_verified=SAT | llm_yes=SAT) and plot a 4x3 figure ====================

def _is_sat_str(x):
    # Treat only the string 'SAT' as SAT; all other values are False
    return str(x).strip().upper() == "SAT"

# 1) Select required columns and compute the conditional probability by (model, N_meta)
_per_cols = ["model", "N_meta", "ground_truth", "pred_llm_yes", "pred_assignment_verified"]
_per_raw = PerSample_all_model_all_N[_per_cols].copy()

_per_raw["llm_yes_is_sat"] = _per_raw["pred_llm_yes"].apply(_is_sat_str)
_per_raw["assign_is_sat"] = _per_raw["pred_assignment_verified"].apply(_is_sat_str)

def _agg_conditional(g):
    denom = int(g["llm_yes_is_sat"].sum())
    num = int(((g["llm_yes_is_sat"]) & (g["assign_is_sat"])).sum())
    rate = (num / denom) if denom > 0 else np.nan
    return pd.Series({"yes_total": denom, "valid_yes_total": num, "assign_given_llm_sat": rate})

assign_given_sat_df = (
    _per_raw
    .groupby(["model", "N_meta"], dropna=False)
    .apply(_agg_conditional)
    .reset_index()
    .rename(columns={"N_meta": "N"})
)

# 2) Prepare model order and x-axis for the new figure
MODELS_4x3 = ordered_models(pd.concat([
    pred_llm_yes_ACC['model'], pred_SAT['model'], pred_UNSAT['model'],
    pred_ADR['model'], pred_MCC['model'],
    assign_given_sat_df['model']
], ignore_index=True))

X_ORDER_4x3 = sorted(set(pd.concat([
    pred_llm_yes_ACC['N'], pred_SAT['N'], pred_UNSAT['N'],
    pred_ADR['N'], pred_MCC['N'],
    assign_given_sat_df['N']
], ignore_index=True).dropna().unique()))
xpos_map_4x3 = {v: i for i, v in enumerate(X_ORDER_4x3)}
xpos_ticks_4x3 = list(range(len(X_ORDER_4x3)))

# 3) Create the 4x3 figure and panel definitions
fig4x3, axes4x3 = plt.subplots(4, 3, figsize=(22, 20))
axes4x3 = axes4x3.ravel()

panels_4x3 = [
    # Row 1: SAT
    ("(a) Precision (SAT)", pred_SAT, "precision"),
    ("(b) Recall (SAT)", pred_SAT, "recall"),
    ("(c) F1 (SAT)", pred_SAT, "f1"),

    # Row 2: UNSAT
    ("(d) Precision (UNSAT)", pred_UNSAT, "precision"),
    ("(e) Recall (UNSAT)", pred_UNSAT, "recall"),
    ("(f) F1 (UNSAT)", pred_UNSAT, "f1"),

    # Row 3
    ("(g) Accuracy", pred_llm_yes_ACC, "accuracy"),
    ("(h) ADR", pred_ADR, "ADR"),
    ("(i) MCC", pred_MCC, "MCC"),

    # Row 4
    ("(j) Assign Verified | LLM=SAT", assign_given_sat_df, "assign_given_llm_sat"),
    (None, None, None),
    (None, None, None),
]

def _panel_ylim_4x3(value_col: str):
    return (-1.05, 1.05) if (value_col and value_col.upper() == "MCC") else (0, 1.05)

def _plot_panel_4x3(ax, title, df_src, value_col: str):
    if title is None:
        ax.axis("off")
        return

    ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_ylabel(value_col.upper() if value_col else "")
    ax.set_xticks(xpos_ticks_4x3)
    ax.set_xticklabels(X_ORDER_4x3)
    ax.set_xlim(-0.2, len(X_ORDER_4x3) - 0.8)
    ymin, ymax = _panel_ylim_4x3(value_col)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linewidth=1.1)

    if df_src is None or df_src.empty or (value_col not in df_src.columns):
        return

    for m in MODELS_4x3:
        st = style_for(m)
        if st is None:
            continue
        _, color, ls, marker = st
        sub = df_src[df_src['model'].astype(str).str.lower() == m.lower()].copy()
        if sub.empty:
            continue
        sub = sub[sub['N'].isin(X_ORDER_4x3)]
        if sub.empty:
            continue
        sub['xpos'] = sub['N'].map(xpos_map_4x3)
        sub = sub.sort_values('xpos')
        y = pd.to_numeric(sub[value_col], errors='coerce')
        ax.plot(
            sub['xpos'], y,
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.6,
            label=legend_label(m)
        )

for ax, (title, df_src, col) in zip(axes4x3, panels_4x3):
    _plot_panel_4x3(ax, title, df_src, col)

# 4) Bottom legend for the 4x3 figure
handles4, labels4, seen4 = [], [], set()
for m in MODELS_4x3:
    st = style_for(m)
    if st is None:
        continue
    _, color, ls, marker = st
    lab = legend_label(m)
    if lab in seen4:
        continue
    seen4.add(lab)
    h = Line2D([0], [0], color=color, linestyle=ls, marker=marker,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.6, label=lab)
    handles4.append(h)
    labels4.append(lab)

fig4x3.legend(handles=handles4, labels=labels4,
              loc="lower center", ncol=4, frameon=True,
              columnspacing=1.4, handletextpad=0.6, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig4x3.subplots_adjust(hspace=0.38, wspace=0.30)

# 5) Save the 4x3 figure
pdf_path_4x3 = SAVE_BASE_4x3 + ".pdf"
svg_path_4x3 = SAVE_BASE_4x3 + ".svg"
png_path_4x3 = SAVE_BASE_4x3 + ".png"

fig4x3.savefig(pdf_path_4x3, dpi=600, bbox_inches="tight")
fig4x3.savefig(svg_path_4x3, dpi=600, bbox_inches="tight")
fig4x3.savefig(png_path_4x3, dpi=600, bbox_inches="tight")

print("Saved PDF (4x3):", pdf_path_4x3)
print("Saved SVG (4x3):", svg_path_4x3)
print("Saved PNG (4x3):", png_path_4x3)


def build_merged_metrics_4x3(
    pred_SAT, pred_UNSAT, pred_llm_yes_ACC, pred_ADR, pred_MCC, assign_given_sat_df,
    export_path=None, filter_with_models=True
):
    """
    Merge all 4x3 subplot sources horizontally on (model, N).

    Output columns:
      model, N,
      precision_sat, recall_sat, f1_sat,
      precision_unsat, recall_unsat, f1_unsat,
      accuracy, ADR, MCC,
      assign_given_llm_sat

    Parameters
    ----------
    pred_SAT : pd.DataFrame
        Columns: ['model', 'N', 'precision', 'recall', 'f1']
    pred_UNSAT : pd.DataFrame
        Columns: ['model', 'N', 'precision', 'recall', 'f1']
    pred_llm_yes_ACC : pd.DataFrame
        Columns: ['model', 'N', 'accuracy']
    pred_ADR : pd.DataFrame
        Columns: ['model', 'N', 'ADR']
    pred_MCC : pd.DataFrame
        Columns: ['model', 'N', 'MCC']
    assign_given_sat_df : pd.DataFrame
        Columns: ['model', 'N', 'assign_given_llm_sat']
    export_path : str or None
        If provided, save the merged table to this Excel path.
    filter_with_models : bool
        If True and global MODELS_4x3 / X_ORDER_4x3 exist, filter rows to the values
        actually drawn in the 4x3 figure.

    Returns
    -------
    pd.DataFrame
    """
    def _safe_pick(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        have = [c for c in cols if c in df.columns]
        if not have:
            return pd.DataFrame(columns=cols)
        return df[have].copy()

    # SAT metrics
    sat_src = _safe_pick(pred_SAT, ["model", "N", "precision", "recall", "f1"])
    sat = sat_src.rename(columns={
        "precision": "precision_sat",
        "recall": "recall_sat",
        "f1": "f1_sat",
    })

    # UNSAT metrics
    unsat_src = _safe_pick(pred_UNSAT, ["model", "N", "precision", "recall", "f1"])
    unsat = unsat_src.rename(columns={
        "precision": "precision_unsat",
        "recall": "recall_unsat",
        "f1": "f1_unsat",
    })

    # Accuracy, ADR, MCC, assignment conditional rate
    acc = _safe_pick(pred_llm_yes_ACC, ["model", "N", "accuracy"])
    adr = _safe_pick(pred_ADR, ["model", "N", "ADR"])
    mcc = _safe_pick(pred_MCC, ["model", "N", "MCC"])
    alig = _safe_pick(assign_given_sat_df, ["model", "N", "assign_given_llm_sat"])

    # Outer merge on (model, N)
    merged = sat.merge(unsat, on=["model", "N"], how="outer") \
                .merge(acc, on=["model", "N"], how="outer") \
                .merge(adr, on=["model", "N"], how="outer") \
                .merge(mcc, on=["model", "N"], how="outer") \
                .merge(alig, on=["model", "N"], how="outer")

    if filter_with_models:
        try:
            models_norm = [m.lower() for m in MODELS_4x3]
            merged = merged[merged["model"].astype(str).str.lower().isin(models_norm)]
            merged = merged[merged["N"].isin(X_ORDER_4x3)]
        except NameError:
            pass

    merged = merged.sort_values(by=["model", "N"]).reset_index(drop=True)

    if export_path:
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        except Exception:
            pass
        merged.to_excel(export_path, index=False)
        print("Merged metrics Excel:", export_path)

    return merged

# Example call
merged_4x3 = build_merged_metrics_4x3(
    pred_SAT, pred_UNSAT, pred_llm_yes_ACC, pred_ADR, pred_MCC, assign_given_sat_df,
    export_path=SAVE_BASE_4x3 + "_merged_metrics.xlsx"
)