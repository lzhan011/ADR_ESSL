import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 路径（按你给的） =========
root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
three_level_similar = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result'
three_level_similar_analysis_dir = os.path.join(three_level_similar, 'analysis')
four_level_similar_analysis_dir = os.path.join(three_level_similar_analysis_dir, 'three_ways_evaluation')

ALL_PAIRS_XLSX_PATH = os.path.join(
    four_level_similar_analysis_dir, "pair_predictions_all_models.xlsx"
)

OUT_DIR = os.path.join(three_level_similar, 'analysis/irt_outputs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "metrics_by_level.xlsx")
OUT_PNG  = os.path.join(OUT_DIR, "metrics_grid_pair_vs_instance.png")

# ========= 工具函数 =========
def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y","t"])

def _safe_div(num, den):
    num = float(num); den = float(den)
    return num / den if den != 0 else 0.0

def _metrics_from_confusion(tp, fp, tn, fn):
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    f1   = _safe_div(2 * prec * rec, prec + rec)
    acc  = _safe_div(tp + tn, tp + tn + fp + fn)   # 新增 Accuracy
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn) / mcc_den) if mcc_den != 0 else 0.0
    return prec, rec, f1, acc, mcc

# ========= 读入 & 规范 =========
df = pd.read_excel(ALL_PAIRS_XLSX_PATH)

needed = ["model","N","original_file","fixed_file","original_prediction","fixed_prediction","is_satisfied"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"输入文件缺少列: {missing}\n现有列: {list(df.columns)}")

df["original_prediction"] = _to_bool(df["original_prediction"])
df["fixed_prediction"]    = _to_bool(df["fixed_prediction"])
df["is_satisfied"]        = _to_bool(df["is_satisfied"])

# ========= Pair-level 指标 =========
df_pair = df.copy()
df_pair["pred_pair_pos"]  = (~df_pair["original_prediction"]) & (df_pair["fixed_prediction"])
df_pair["label_pair_pos"] = df_pair["is_satisfied"]  # 真值：是否满足
df_pair["y_strict"]       = df_pair["pred_pair_pos"] & df_pair["is_satisfied"]  # ADR-Strict 计分

def agg_pair(grp: pd.DataFrame):
    y_true = grp["label_pair_pos"].astype(int).to_numpy()
    y_pred = grp["pred_pair_pos"].astype(int).to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec, rec, f1, acc, mcc = _metrics_from_confusion(tp, fp, tn, fn)
    adr = float(grp["y_strict"].mean())  # ADR-Strict

    return pd.Series({
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "mcc": mcc,
        "ADR": adr, "n_pairs": len(grp)
    })

pair_metrics = (
    df_pair
    .groupby(["model","N"], as_index=False)
    .apply(agg_pair)
    .reset_index(drop=True)
)

# ========= Instance-level 指标 =========
# 仅使用 is_satisfied==True 的 pair 来构造真值
df_ok = df[df["is_satisfied"]].copy()

inst_left = pd.DataFrame({
    "model": df_ok["model"],
    "N": df_ok["N"],
    "item_id": df_ok["original_file"].astype(str),
    "truth": False,
    "pred": df_ok["original_prediction"].astype(bool)
})
inst_right = pd.DataFrame({
    "model": df_ok["model"],
    "N": df_ok["N"],
    "item_id": df_ok["fixed_file"].astype(str),
    "truth": True,
    "pred": df_ok["fixed_prediction"].astype(bool)
})
df_inst = pd.concat([inst_left, inst_right], ignore_index=True)
df_inst["correct"] = (df_inst["pred"] == df_inst["truth"])   # instance 级 ADR = 平均正确率

def agg_inst(grp: pd.DataFrame):
    y_true = grp["truth"].astype(int).to_numpy()
    y_pred = grp["pred"].astype(int).to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec, rec, f1, acc, mcc = _metrics_from_confusion(tp, fp, tn, fn)
    adr = float(grp["correct"].mean())

    return pd.Series({
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "mcc": mcc,
        "ADR": adr, "n_instances": len(grp)
    })

instance_metrics = (
    df_inst
    .groupby(["model","N"], as_index=False)
    .apply(agg_inst)
    .reset_index(drop=True)
)

# ========= 写到一个 Excel 的两个 sheet =========
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    pair_metrics.sort_values(["model","N"]).to_excel(writer, sheet_name="pair_level", index=False)
    instance_metrics.sort_values(["model","N"]).to_excel(writer, sheet_name="instance_level", index=False)

print(f"[OK] 指标已写入：{OUT_XLSX}")

# ========= 4×3 指标总览图 =========
def _plot_grouped_bars(ax, df_metric, metric_name, title):
    """
    df_metric: 包含 ['model','N', <metric_name>] 三列
    在 ax 上绘制：x= model；每个 N 一个颜色并列柱
    """
    # 模型顺序固定（按名称）
    models = sorted(df_metric["model"].unique().tolist())
    Ns = sorted(df_metric["N"].dropna().unique().tolist())
    x = np.arange(len(models))
    if len(Ns) == 0:
        Ns = [None]
    width = 0.8 / max(len(Ns), 1)

    for i, N in enumerate(Ns):
        if N is None:
            sub = df_metric.copy()
            label = "N=NA"
        else:
            sub = df_metric[df_metric["N"] == N]
            label = f"N={N}"

        y = []
        for m in models:
            v = sub.loc[sub["model"] == m, metric_name]
            y.append(float(v.iloc[0]) if len(v) else np.nan)

        ax.bar(x + (i - (len(Ns)-1)/2)*width, y, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim(0, 1)  # 常见指标范围
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

# 准备绘图数据
inst_df = instance_metrics.copy()
pair_df = pair_metrics.copy()

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
plt.subplots_adjust(hspace=0.35, wspace=0.18)

# 第 1 行（instance：P / R / F1）
_plot_grouped_bars(axes[0,0], inst_df, "precision", "Instance — Precision")
_plot_grouped_bars(axes[0,1], inst_df, "recall",    "Instance — Recall")
_plot_grouped_bars(axes[0,2], inst_df, "f1",        "Instance — F1")

# 第 2 行（instance：Acc / MCC / ADR）
_plot_grouped_bars(axes[1,0], inst_df, "accuracy",  "Instance — Accuracy")
_plot_grouped_bars(axes[1,1], inst_df, "mcc",       "Instance — MCC")
_plot_grouped_bars(axes[1,2], inst_df, "ADR",       "Instance — ADR")

# 第 3 行（pair：P / R / F1）
_plot_grouped_bars(axes[2,0], pair_df, "precision", "Pair — Precision")
_plot_grouped_bars(axes[2,1], pair_df, "recall",    "Pair — Recall")
_plot_grouped_bars(axes[2,2], pair_df, "f1",        "Pair — F1")

# 第 4 行（pair：Acc / MCC / ADR）
_plot_grouped_bars(axes[3,0], pair_df, "accuracy",  "Pair — Accuracy")
_plot_grouped_bars(axes[3,1], pair_df, "mcc",       "Pair — MCC")
_plot_grouped_bars(axes[3,2], pair_df, "ADR",       "Pair — ADR")

# 统一图例（放在顶部）
handles, labels = axes[0,0].get_legend_handles_labels()
if labels:
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"[OK] 指标总览图已保存：{OUT_PNG}")











# ========= 计算“不考虑 N”的总指标（按 model 聚合） =========
def _agg_from_bool_arrays(y_true_bool, y_pred_bool):
    y_true = y_true_bool.astype(int).to_numpy()
    y_pred = y_pred_bool.astype(int).to_numpy()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec, rec, f1, acc, mcc = _metrics_from_confusion(tp, fp, tn, fn)
    return tp, fp, tn, fn, prec, rec, f1, acc, mcc

# ---- pair overall ----
pair_overall_rows = []
for m, g in df_pair.groupby("model"):
    tp, fp, tn, fn, prec, rec, f1, acc, mcc = _agg_from_bool_arrays(
        g["label_pair_pos"], g["pred_pair_pos"]
    )
    adr = float((g["pred_pair_pos"] & g["is_satisfied"]).mean())  # ADR-Strict 的平均
    pair_overall_rows.append({
        "model": m, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "mcc": mcc,
        "ADR": adr, "n_pairs": len(g)
    })
pair_overall = pd.DataFrame(pair_overall_rows).sort_values("model")

# ---- instance overall ----
inst_overall_rows = []
for m, g in df_inst.groupby("model"):
    tp, fp, tn, fn, prec, rec, f1, acc, mcc = _agg_from_bool_arrays(
        g["truth"], g["pred"]
    )
    adr = float(g["correct"].mean())  # instance 的 ADR=准确率
    inst_overall_rows.append({
        "model": m, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "mcc": mcc,
        "ADR": adr, "n_instances": len(g)
    })
instance_overall = pd.DataFrame(inst_overall_rows).sort_values("model")

# ========= 写入一个 Excel：新增 overall 的两个 sheet =========
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    pair_metrics.sort_values(["model","N"]).to_excel(writer, sheet_name="pair_level", index=False)
    instance_metrics.sort_values(["model","N"]).to_excel(writer, sheet_name="instance_level", index=False)
    pair_overall.to_excel(writer, sheet_name="pair_overall", index=False)
    instance_overall.to_excel(writer, sheet_name="instance_overall", index=False)
print(f"[OK] 指标已写入：{OUT_XLSX}")

# ========= 画“不分 N”的四指标总览图（2×2） =========
def plot_overall_four(df_overall: pd.DataFrame, level_name: str, out_path: str):
    """
    画 2×2：Accuracy / F1 / MCC / ADR（y 轴统一到 [0,1]，MCC 轴为 [-1,1]）
    """
    models = df_overall["model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Accuracy
    ax = axes[0, 0]
    ax.bar(models, df_overall["accuracy"].values)
    ax.set_title(f"{level_name} — Accuracy"); ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=35); ax.grid(axis="y", linestyle="--", alpha=0.3)

    # F1
    ax = axes[0, 1]
    ax.bar(models, df_overall["f1"].values)
    ax.set_title(f"{level_name} — F1"); ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=35); ax.grid(axis="y", linestyle="--", alpha=0.3)

    # MCC（范围 -1 到 1）
    ax = axes[1, 0]
    ax.bar(models, df_overall["mcc"].values)
    ax.set_title(f"{level_name} — MCC"); ax.set_ylim(-1, 1)
    ax.tick_params(axis='x', rotation=35); ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ADR
    ax = axes[1, 1]
    ax.bar(models, df_overall["ADR"].values)
    ax.set_title(f"{level_name} — ADR"); ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=35); ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] 已保存：{out_path}")

OUT_PNG_INST_OVERALL = os.path.join(OUT_DIR, "overall_instance_metrics_2x2.png")
OUT_PNG_PAIR_OVERALL = os.path.join(OUT_DIR, "overall_pair_metrics_2x2.png")

plot_overall_four(instance_overall, "Instance (overall)", OUT_PNG_INST_OVERALL)
plot_overall_four(pair_overall,     "Pair (overall)",     OUT_PNG_PAIR_OVERALL)

