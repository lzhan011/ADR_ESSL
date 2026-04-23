import os
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Read the Excel file
file_path = os.path.join(BASE_DIR, "..", "Analysis_Result_Collection", "SAT_figures_V20250906.xlsx")
df = pd.read_excel(file_path, sheet_name="UNSAT_Prediction")

# Remove empty rows and normalize column types
df = df.dropna(subset=["model", "N", "correct_rate"]).copy()
df["N"] = pd.to_numeric(df["N"], errors="coerce")
df["correct_rate"] = pd.to_numeric(df["correct_rate"], errors="coerce")

# --- Remove models that should not be shown ---
# 1) Remove all gpt-4-turbo entries
df = df[~df["model"].astype(str).str.contains(r"\bgpt-4-turbo\b", case=False, na=False)].copy()
# 2) Remove only records exactly equal to gpt-4o, while keeping gpt-4o-latest
df = df[df["model"].astype(str).str.strip().str.lower() != "gpt-4o"].copy()

# ---------- Global style ----------
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 16,
    "lines.linewidth": 3.0, "lines.markersize": 10.0,
})

# ---------- Legend text cleanup ----------
def legend_label(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    # Claude: remove date suffixes
    if low.startswith("claude"):
        s = re.sub(r"[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}$", "", s)
        s = re.sub(r"[-_]?20\d{6,8}$", "", s)
    # DeepSeek: remove parentheses and their contents
    if "deepseek" in low:
        s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s

# ---------- Grouping and style mapping ----------
COLOR_CLAUDE = "#7b3fc8"      # Purple
COLOR_DEEPSEEK = "#2ca02c"    # Green
COLOR_OAI_RED = "#d62728"     # Red
COLOR_OAI_YELL = "#ffb000"    # Yellow

LS_CLAUDE = "--"              # Dashed line
LS_DEEPSEEK = (0, (10, 4))      # Long dashed line
LS_OPENAI = "-"               # Solid line

DEEPSEEK_SET = {
    "deepseek-chat": "*",      # Star
    "deepseek-reasoner": "s",  # Square
}
OPENAI_RED_SET = {                # Red group with solid lines
    "gpt-5": "o",
    "o1": "D",
    "o3-mini": "X",
}

# Stable triangle selection for Claude models
def _claude_marker(ml: str) -> str:
    ml = ml.lower()
    if "haiku" in ml:
        return "^"              # Up triangle
    if "sonnet" in ml and (("3-7" in ml) or ("3.7" in ml)):
        return "v"              # Down triangle
    if ("3-opus" in ml) or (("opus" in ml) and re.search(r"\b3([._-]\d+)?\b", ml)):
        return "<"              # Left triangle
    if re.search(r"(opus|sonnet)[-_ ]?4\b", ml) or "opus-4" in ml:
        return ">"              # Right triangle
    return ["^", "v", "<", ">"][(hash(ml) % 4)]

# Stable marker selection for the yellow OpenAI group
def _openai_yellow_marker(ml: str) -> str:
    ml = ml.lower()
    if re.search(r"\bgpt[-_]?4[.\-]?1\b", ml):
        return "P"              # Pentagon
    if "4o-latest" in ml:       # gpt-4o-latest / chatgpt-4o-latest
        return "h"              # Hexagon
    if re.search(r"gpt[-_]?3\.?5[-_]?turbo", ml):
        return "d"              # Thin diamond
    # Fallback: assign one of P/h/d deterministically
    return ["P", "h", "d"][(hash(ml) % 3)]


def style_for(model: str):
    """Return (group, color, linestyle, marker)."""
    m = str(model).strip()
    ml = m.lower()

    if ml.startswith("claude"):
        return ("claude", COLOR_CLAUDE, LS_CLAUDE, _claude_marker(ml))

    if "deepseek-chat" in ml:
        return ("deepseek", COLOR_DEEPSEEK, LS_DEEPSEEK, "*")
    if "deepseek-reasoner" in ml:
        return ("deepseek", COLOR_DEEPSEEK, LS_DEEPSEEK, "s")

    if ml in OPENAI_RED_SET:
        return ("openai-red", COLOR_OAI_RED, LS_OPENAI, OPENAI_RED_SET[ml])

    # Other OpenAI models go to the yellow group
    if any(x in ml for x in ["gpt", "o1", "o3", "openai", "chatgpt"]):
        return ("openai-yellow", COLOR_OAI_YELL, LS_OPENAI, _openai_yellow_marker(ml))

    # Fallback
    return ("other", "#444444", "-", "o")


# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 6))

# 1) Build an equally spaced categorical x-axis that always shows 5/8/10,
#    then append any additional N values found in the data
must_show = [5, 8, 10]
present = sorted(set(pd.to_numeric(df["N"], errors="coerce").dropna()))
N_ORDER = must_show + [n for n in present if n not in must_show]   # e.g. [5,8,10,25,50,60,90,110,140,...]
pos_map = {n: i for i, n in enumerate(N_ORDER)}                    # N -> equally spaced position 0..K-1

GROUP_ORDER = ["claude", "deepseek", "openai-red", "openai-yellow", "other"]
groups = {g: [] for g in GROUP_ORDER}

for model, subdf in df.groupby("model"):
    g, color, ls, marker = style_for(model)
    groups.setdefault(g, []).append((model, subdf.copy(), color, ls, marker))

for g in GROUP_ORDER:
    for model, subdf, color, ls, marker in groups.get(g, []):
        subdf = subdf.sort_values("N")
        x_pos = subdf["N"].map(pos_map)   # Plot using equally spaced positions
        ax.plot(
            x_pos, subdf["correct_rate"],
            label=legend_label(model),
            color=color, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgecolor="black",   # Hollow marker with black edge
            markeredgewidth=1.6
        )

# 2) Set equally spaced ticks and labels, ensuring 5/8/10 are always shown
ax.set_xticks(range(len(N_ORDER)))
ax.set_xticklabels([str(n) for n in N_ORDER])
ax.set_xlim(-0.5, len(N_ORDER) - 0.5)

ax.set_xlabel("N")
ax.set_ylabel("Correct Rate")
# Keep only horizontal grid lines to avoid suggesting a linear x-axis distance
ax.grid(True, axis="y", linewidth=1.0, alpha=0.7)

# Place a vertical legend on the right side of the axes, centered vertically
legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),   # Slightly outside the right side of the axes
    ncol=1,                       # Vertical layout
    frameon=True,
    borderaxespad=0.0,
    columnspacing=1.0,
    handletextpad=0.6,
)

# Reserve space for the right-side legend and slightly lower the bottom margin
plt.subplots_adjust(left=0.10, right=0.72, bottom=0.12)

# Save outputs
out_dir = os.path.join(
    BASE_DIR,
    "..",
    "Analysis_Result_Collection",
    "Figure_in_paper",
    "unsat"
)
os.makedirs(out_dir, exist_ok=True)
outbase = os.path.join(out_dir, "unsat_small_alpha_prediction_correct_rate")
fig.savefig(outbase + ".pdf", bbox_inches="tight")
fig.savefig(outbase + ".svg", bbox_inches="tight")
fig.savefig(outbase + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("The figure has been saved. The x-axis uses equally spaced categorical positions with 5/8/10 forced to appear, and the legend is placed on the right.")
print(f"Output path: {os.path.abspath(outbase)}")
