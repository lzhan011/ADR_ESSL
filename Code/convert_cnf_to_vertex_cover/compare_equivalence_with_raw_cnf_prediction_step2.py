import os
import pandas as pd

# 只保留图中出现的模型
MODEL_WHITELIST = {
    "deepseek-reasoner",
    "o3-mini",
    "o1",
    "gpt-5",
    "gpt-3.5-turbo-0125",
    "gpt-4.1",
    "chatgpt-4o-latest",
}

def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    把常见 True/False/1/0/yes/no/sat/unsat 转成 pandas 可空布尔类型
    """
    if str(series.dtype) == "boolean":
        return series
    if series.dtype == bool:
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda x: None if pd.isna(x) else bool(int(x))).astype("boolean")

    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "sat": True, "unsat": False,
    }

    def _conv(x):
        if pd.isna(x):
            return None
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return mapping.get(s, None)

    return series.map(_conv).astype("boolean")


def _sum_bool(s: pd.Series) -> int:
    return int(s.fillna(False).sum())


def _safe_ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def compute_agree_disagree_with_label_metrics(input_xlsx: str, output_dir: str = None, sheet_name=0):
    df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

    required_cols = ["model", "N", "cnf_label_IS_SAT", "cnf_prediction_IS_SAT", "VC_llm_answer_yes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 统一布尔类型
    for c in ["cnf_label_IS_SAT", "cnf_prediction_IS_SAT", "VC_llm_answer_yes"]:
        df[c] = to_nullable_bool(df[c])

    # 仅保留可比较行
    valid = df[
        df["cnf_label_IS_SAT"].notna()
        & df["cnf_prediction_IS_SAT"].notna()
        & df["VC_llm_answer_yes"].notna()
    ].copy()

    # 基础关系
    valid["cnf_vc_agree"] = (valid["cnf_prediction_IS_SAT"] == valid["VC_llm_answer_yes"]).astype("boolean")
    valid["cnf_vc_disagree"] = (valid["cnf_prediction_IS_SAT"] != valid["VC_llm_answer_yes"]).astype("boolean")

    valid["cnf_correct_vs_label"] = (valid["cnf_prediction_IS_SAT"] == valid["cnf_label_IS_SAT"]).astype("boolean")
    valid["vc_correct_vs_label"] = (valid["VC_llm_answer_yes"] == valid["cnf_label_IS_SAT"]).astype("boolean")

    # agree 子集里的“共同预测是否与 label 一致”
    # 因为 agree 时 CNF=VC，所以看其中一个即可（这里用 CNF）
    valid["agree_and_match_label"] = (
        (valid["cnf_vc_agree"] == True) & (valid["cnf_correct_vs_label"] == True)
    ).astype("boolean")

    valid["agree_and_mismatch_label"] = (
        (valid["cnf_vc_agree"] == True) & (valid["cnf_correct_vs_label"] == False)
    ).astype("boolean")

    # disagree 子集里的“谁对”
    valid["disagree_and_cnf_correct"] = (
        (valid["cnf_vc_disagree"] == True) & (valid["cnf_correct_vs_label"] == True)
    ).astype("boolean")

    valid["disagree_and_vc_correct"] = (
        (valid["cnf_vc_disagree"] == True) & (valid["vc_correct_vs_label"] == True)
    ).astype("boolean")

    valid["disagree_and_both_correct"] = (
        (valid["cnf_vc_disagree"] == True)
        & (valid["cnf_correct_vs_label"] == True)
        & (valid["vc_correct_vs_label"] == True)
    ).astype("boolean")

    valid["disagree_and_both_wrong"] = (
        (valid["cnf_vc_disagree"] == True)
        & (valid["cnf_correct_vs_label"] == False)
        & (valid["vc_correct_vs_label"] == False)
    ).astype("boolean")

    # 仅保留图中模型（白名单过滤）
    before_rows = len(valid)
    valid = valid[valid["model"].astype(str).str.strip().isin(MODEL_WHITELIST)].copy()
    after_rows = len(valid)

    print(f"[INFO] 模型白名单过滤: {before_rows} -> {after_rows}")
    print(f"[INFO] 保留模型: {sorted(valid['model'].astype(str).unique().tolist())}")

    if valid.empty:
        raise RuntimeError("白名单过滤后 valid 为空，请检查模型名称是否一致（例如 gpt-4o-latest vs chatgpt-4o-latest）。")

    def summarize_group(g: pd.DataFrame) -> pd.Series:
        total_valid = int(len(g))

        agree_mask = (g["cnf_vc_agree"] == True)
        disagree_mask = (g["cnf_vc_disagree"] == True)

        agree_count = int(agree_mask.sum())
        disagree_count = int(disagree_mask.sum())

        # agree 子集统计
        agree_match_cnt = _sum_bool(g.loc[agree_mask, "cnf_correct_vs_label"])
        # 在 agree 子集里，不一致标签数就是 agree_count - agree_match_cnt
        agree_mismatch_cnt = agree_count - agree_match_cnt

        # disagree 子集统计
        disagree_cnf_correct_cnt = _sum_bool(g.loc[disagree_mask, "cnf_correct_vs_label"])
        disagree_vc_correct_cnt = _sum_bool(g.loc[disagree_mask, "vc_correct_vs_label"])
        disagree_both_correct_cnt = _sum_bool(g.loc[disagree_mask, "disagree_and_both_correct"])
        disagree_both_wrong_cnt = _sum_bool(g.loc[disagree_mask, "disagree_and_both_wrong"])

        out = {
            # 基础计数
            "valid_rows_count": total_valid,
            "agree_count": agree_count,
            "disagree_count": disagree_count,

            # ===== 你新增要的（agree 子集）=====
            "agree_match_label_count": agree_match_cnt,
            "agree_mismatch_label_count": agree_mismatch_cnt,
            "agree_match_label_ratio": _safe_ratio(agree_match_cnt, agree_count),
            "agree_mismatch_label_ratio": _safe_ratio(agree_mismatch_cnt, agree_count),

            # ===== 原来要的（disagree 子集）=====
            "cnf_correct_when_disagree_count": disagree_cnf_correct_cnt,
            "cnf_correct_when_disagree_ratio": _safe_ratio(disagree_cnf_correct_cnt, disagree_count),

            "vc_correct_when_disagree_count": disagree_vc_correct_cnt,
            "vc_correct_when_disagree_ratio": _safe_ratio(disagree_vc_correct_cnt, disagree_count),

            # 辅助诊断（可选）
            "both_correct_when_disagree_count": disagree_both_correct_cnt,
            "both_wrong_when_disagree_count": disagree_both_wrong_cnt,
            "both_correct_when_disagree_ratio": _safe_ratio(disagree_both_correct_cnt, disagree_count),
            "both_wrong_when_disagree_ratio": _safe_ratio(disagree_both_wrong_cnt, disagree_count),
        }
        return pd.Series(out)

    # 分组汇总（你要的主表：拼接后的结果）
    summary_by_model_N = (
        valid.groupby(["model", "N"], dropna=False)
        .apply(summarize_group)
        .reset_index()
        .sort_values(["model", "N"], kind="mergesort")
        .reset_index(drop=True)
    )

    # 总体汇总
    overall = summarize_group(valid).to_frame().T
    overall.insert(0, "scope", "ALL")

    # 明细（可选导出）
    agree_details = valid[valid["cnf_vc_agree"] == True].copy()
    disagree_details = valid[valid["cnf_vc_disagree"] == True].copy()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_xlsx = os.path.join(output_dir, "cnf_vc_agree_disagree_with_label_metrics.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            summary_by_model_N.to_excel(writer, sheet_name="by_model_N", index=False)
            overall.to_excel(writer, sheet_name="overall", index=False)
            agree_details.to_excel(writer, sheet_name="agree_details", index=False)
            disagree_details.to_excel(writer, sheet_name="disagree_details", index=False)
        print(f"Saved: {out_xlsx}")

    return summary_by_model_N, overall, agree_details, disagree_details


import pandas as pd


def _safe_div(num, den):
    return (num / den) if den else 0.0


def summarize_agree_disagree_from_overall(xlsx_path: str, sheet_name: str = "overall") -> dict:
    """
    从 cnf_vc_agree_disagree_with_label_metrics.xlsx 的 overall sheet 读取计数并计算比例。

    返回字典包含：
      - counts
      - ratios (0-1)
      - percents (0-100)
    """
    overall = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if overall.empty:
        raise ValueError(f"{sheet_name} sheet is empty: {xlsx_path}")

    row = overall.iloc[0]

    # 读取计数
    valid_rows_count = int(row["valid_rows_count"])
    agree_count = int(row["agree_count"])
    disagree_count = int(row["disagree_count"])

    agree_match_label_count = int(row["agree_match_label_count"])
    agree_mismatch_label_count = int(row["agree_mismatch_label_count"])

    cnf_correct_when_disagree_count = int(row["cnf_correct_when_disagree_count"])

    # 兼容 VC 或 packing 列名
    if "vc_correct_when_disagree_count" in row.index:
        counterpart_col = "vc_correct_when_disagree_count"
        counterpart_name = "vc"
    elif "packing_correct_when_disagree_count" in row.index:
        counterpart_col = "packing_correct_when_disagree_count"
        counterpart_name = "packing"
    else:
        raise KeyError("Neither 'vc_correct_when_disagree_count' nor "
                       "'packing_correct_when_disagree_count' found in overall sheet.")

    counterpart_correct_when_disagree_count = int(row[counterpart_col])

    # 计算比例
    agree_match_label_ratio = _safe_div(agree_match_label_count, agree_count)
    agree_mismatch_label_ratio = _safe_div(agree_mismatch_label_count, agree_count)

    cnf_correct_when_disagree_ratio = _safe_div(cnf_correct_when_disagree_count, disagree_count)
    counterpart_correct_when_disagree_ratio = _safe_div(counterpart_correct_when_disagree_count, disagree_count)

    agree_ratio_overall = _safe_div(agree_count, valid_rows_count)
    disagree_ratio_overall = _safe_div(disagree_count, valid_rows_count)

    result = {
        "source": {"xlsx_path": xlsx_path, "sheet_name": sheet_name, "mode": "overall"},
        "counts": {
            "valid_rows_count": valid_rows_count,
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "agree_match_label_count": agree_match_label_count,
            "agree_mismatch_label_count": agree_mismatch_label_count,
            "cnf_correct_when_disagree_count": cnf_correct_when_disagree_count,
            f"{counterpart_name}_correct_when_disagree_count": counterpart_correct_when_disagree_count,
        },
        "ratios": {
            "agree_ratio_overall": round(agree_ratio_overall, 4),
            "disagree_ratio_overall": round(disagree_ratio_overall, 4),
            "agree_match_label_ratio": round(agree_match_label_ratio, 4),
            "agree_mismatch_label_ratio": round(agree_mismatch_label_ratio, 4),
            "cnf_correct_when_disagree_ratio": round(cnf_correct_when_disagree_ratio, 4),
            f"{counterpart_name}_correct_when_disagree_ratio": round(counterpart_correct_when_disagree_ratio, 4),
        },
        "percents": {
            "agree_ratio_overall_pct": round(agree_ratio_overall * 100, 2),
            "disagree_ratio_overall_pct": round(disagree_ratio_overall * 100, 2),
            "agree_match_label_ratio_pct": round(agree_match_label_ratio * 100, 2),
            "agree_mismatch_label_ratio_pct": round(agree_mismatch_label_ratio * 100, 2),
            "cnf_correct_when_disagree_ratio_pct": round(cnf_correct_when_disagree_ratio * 100, 2),
            f"{counterpart_name}_correct_when_disagree_ratio_pct": round(counterpart_correct_when_disagree_ratio * 100, 2),
        },
        "counterpart_name": counterpart_name,  # "vc" or "packing"
    }
    return result


def summarize_agree_disagree_from_details(
    xlsx_path: str,
    agree_sheet: str = "agree_details",
    disagree_sheet: str = "disagree_details"
) -> dict:
    """
    从明细 sheet 重算指标（用于交叉验证）。
    兼容列名：
      - cnf_correct_vs_label
      - vc_correct_vs_label 或 packing_correct_vs_label
    """
    agree_df = pd.read_excel(xlsx_path, sheet_name=agree_sheet)
    disagree_df = pd.read_excel(xlsx_path, sheet_name=disagree_sheet)

    def to_bool(s: pd.Series) -> pd.Series:
        if str(s.dtype) == "boolean":
            return s.fillna(False).astype(bool)
        if s.dtype == bool:
            return s.fillna(False)
        return s.astype(str).str.strip().str.lower().isin(
            ["true", "1", "1.0", "yes", "y", "t"]
        )

    if "cnf_correct_vs_label" not in agree_df.columns:
        raise KeyError(f"'cnf_correct_vs_label' not found in {agree_sheet}")

    if "cnf_correct_vs_label" not in disagree_df.columns:
        raise KeyError(f"'cnf_correct_vs_label' not found in {disagree_sheet}")

    agree_df = agree_df.copy()
    disagree_df = disagree_df.copy()

    agree_df["cnf_correct_vs_label"] = to_bool(agree_df["cnf_correct_vs_label"])
    disagree_df["cnf_correct_vs_label"] = to_bool(disagree_df["cnf_correct_vs_label"])

    if "vc_correct_vs_label" in disagree_df.columns:
        counterpart_col = "vc_correct_vs_label"
        counterpart_name = "vc"
    elif "packing_correct_vs_label" in disagree_df.columns:
        counterpart_col = "packing_correct_vs_label"
        counterpart_name = "packing"
    else:
        raise KeyError("Neither 'vc_correct_vs_label' nor 'packing_correct_vs_label' found in disagree_details.")

    disagree_df[counterpart_col] = to_bool(disagree_df[counterpart_col])

    agree_count = len(agree_df)
    disagree_count = len(disagree_df)
    valid_rows_count = agree_count + disagree_count

    agree_match_label_count = int(agree_df["cnf_correct_vs_label"].sum())
    agree_mismatch_label_count = agree_count - agree_match_label_count

    cnf_correct_when_disagree_count = int(disagree_df["cnf_correct_vs_label"].sum())
    counterpart_correct_when_disagree_count = int(disagree_df[counterpart_col].sum())

    agree_match_label_ratio = _safe_div(agree_match_label_count, agree_count)
    agree_mismatch_label_ratio = _safe_div(agree_mismatch_label_count, agree_count)
    cnf_correct_when_disagree_ratio = _safe_div(cnf_correct_when_disagree_count, disagree_count)
    counterpart_correct_when_disagree_ratio = _safe_div(counterpart_correct_when_disagree_count, disagree_count)

    agree_ratio_overall = _safe_div(agree_count, valid_rows_count)
    disagree_ratio_overall = _safe_div(disagree_count, valid_rows_count)

    result = {
        "source": {"xlsx_path": xlsx_path, "agree_sheet": agree_sheet, "disagree_sheet": disagree_sheet, "mode": "details"},
        "counts": {
            "valid_rows_count": valid_rows_count,
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "agree_match_label_count": agree_match_label_count,
            "agree_mismatch_label_count": agree_mismatch_label_count,
            "cnf_correct_when_disagree_count": cnf_correct_when_disagree_count,
            f"{counterpart_name}_correct_when_disagree_count": counterpart_correct_when_disagree_count,
        },
        "ratios": {
            "agree_ratio_overall": round(agree_ratio_overall, 4),
            "disagree_ratio_overall": round(disagree_ratio_overall, 4),
            "agree_match_label_ratio": round(agree_match_label_ratio, 4),
            "agree_mismatch_label_ratio": round(agree_mismatch_label_ratio, 4),
            "cnf_correct_when_disagree_ratio": round(cnf_correct_when_disagree_ratio, 4),
            f"{counterpart_name}_correct_when_disagree_ratio": round(counterpart_correct_when_disagree_ratio, 4),
        },
        "percents": {
            "agree_ratio_overall_pct": round(agree_ratio_overall * 100, 2),
            "disagree_ratio_overall_pct": round(disagree_ratio_overall * 100, 2),
            "agree_match_label_ratio_pct": round(agree_match_label_ratio * 100, 2),
            "agree_mismatch_label_ratio_pct": round(agree_mismatch_label_ratio * 100, 2),
            "cnf_correct_when_disagree_ratio_pct": round(cnf_correct_when_disagree_ratio * 100, 2),
            f"{counterpart_name}_correct_when_disagree_ratio_pct": round(counterpart_correct_when_disagree_ratio * 100, 2),
        },
        "counterpart_name": counterpart_name,
    }
    return result


def format_summary_text(summary: dict) -> str:
    """
    把 summarize_* 返回的字典格式化成英文结果段落。
    """
    c = summary["counts"]
    p = summary["percents"]
    counterpart = summary["counterpart_name"].upper()  # VC / PACKING

    # 先构造动态 key，避免嵌套 f-string 引号冲突
    counterpart_pct_key = f"{summary['counterpart_name']}_correct_when_disagree_ratio_pct"
    counterpart_pct = p[counterpart_pct_key]

    text = (
        f"Across all matched instances, agreement is more common than disagreement "
        f"({c['agree_count']}/{c['valid_rows_count']}, {p['agree_ratio_overall_pct']:.2f}% vs. "
        f"{c['disagree_count']}/{c['valid_rows_count']}, {p['disagree_ratio_overall_pct']:.2f}%). "
        f"Among agreement cases, the shared prediction is correct in "
        f"{p['agree_match_label_ratio_pct']:.2f}% of instances and incorrect in "
        f"{p['agree_mismatch_label_ratio_pct']:.2f}%. "
        f"Among disagreement cases, CNF predictions match the label in "
        f"{p['cnf_correct_when_disagree_ratio_pct']:.2f}% of instances, whereas "
        f"{counterpart} predictions match the label in "
        f"{counterpart_pct:.2f}% of instances."
    )
    return text




if __name__ == "__main__":
    input_xlsx = "/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis/cnf_and_vertex_cover_equivalence/all_file_all_model_all_N_vc_and_cnf_prediction.xlsx"
    output_dir = "/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis/cnf_and_vertex_cover_equivalence"

    by_model_N, overall, agree_details, disagree_details = compute_agree_disagree_with_label_metrics(
        input_xlsx=input_xlsx,
        output_dir=output_dir,
        sheet_name=0,
    )

    print("\n=== by_model_N (拼接后的结果) ===")
    print(by_model_N.head(50))

    print("\n=== overall ===")
    print(overall)

    xlsx_path = os.path.join(output_dir, "cnf_vc_agree_disagree_with_label_metrics.xlsx")
    # 方式1：直接读 overall（推荐）
    summary_overall = summarize_agree_disagree_from_overall(xlsx_path)
    print(summary_overall)
    print()
    print(format_summary_text(summary_overall))

    # 方式2：从明细重算（交叉验证）
    summary_details = summarize_agree_disagree_from_details(xlsx_path)
    print()
    print(summary_details)