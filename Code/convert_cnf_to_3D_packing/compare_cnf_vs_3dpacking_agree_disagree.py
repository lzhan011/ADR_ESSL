import os
import re
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

# =========================
# 基础工具
# =========================
def parse_model_and_n_from_summary_filename(filename: str):
    """
    解析文件名:
      <model>_N<number>_summary_metrics.xlsx
    """
    base = os.path.basename(filename)
    m = re.match(r"(.+)_N(\d+)_summary_metrics\.xlsx$", base, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def canonicalize_filename_for_match(fn: str) -> str:
    """
    为了和 VC 总表匹配，统一文件名格式：
    - .txt -> .cnf
    其余保持不变
    """
    if pd.isna(fn):
        return None
    fn = str(fn).strip()
    if fn.endswith(".txt"):
        return fn[:-4] + ".cnf"
    return fn


def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    把 SAT/UNSAT, True/False, 1/0, yes/no 等转换为 pandas 可空布尔类型
    True 表示 SAT/YES
    False 表示 UNSAT/NO
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
        "1.0": True, "0.0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
        "t": True, "f": False,
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


# =========================
# 读取 3D-packing PerSample 中间文件
# =========================
def load_3dpacking_persample_from_intermediate(intermediate_dir: str) -> pd.DataFrame:
    """
    从 intermediate_results 目录读取所有 *_summary_metrics.xlsx 的 PerSample sheet
    并汇总成一个 DataFrame。
    """
    all_rows = []

    for fname in sorted(os.listdir(intermediate_dir)):
        if not fname.endswith("_summary_metrics.xlsx"):
            continue
        if fname == "summary_metrics.xlsx":
            # 跳过总表（它没有 PerSample 或结构不同/重复）
            continue

        fpath = os.path.join(intermediate_dir, fname)
        model_from_file, n_from_file = parse_model_and_n_from_summary_filename(fname)
        if model_from_file is None:
            continue

        try:
            df = pd.read_excel(fpath, sheet_name="PerSample")
        except Exception as e:
            print(f"[WARN] 读取失败（跳过）: {fpath} | {e}")
            continue

        if df is None or df.empty:
            continue

        # 兼容列名检查
        required = ["filename", "model", "ground_truth", "pred_llm_yes"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] 缺少列（跳过）: {fpath} 缺少 {missing}")
            continue

        # N_meta 有些文件为空；从文件名补
        if "N_meta" not in df.columns:
            df["N_meta"] = n_from_file
        else:
            # 空值用文件名补
            df["N_meta"] = pd.to_numeric(df["N_meta"], errors="coerce")
            df.loc[df["N_meta"].isna(), "N_meta"] = n_from_file

        # model 列有时就正确；若为空则补文件名模型
        df["model"] = df["model"].astype(str).replace("nan", "")
        df.loc[df["model"].str.strip() == "", "model"] = model_from_file

        # 标准化
        keep_cols = [c for c in [
            "filename", "model", "N_meta", "alpha_meta", "ground_truth",
            "pred_llm_yes", "pred_assignment_verified", "source_answer_path"
        ] if c in df.columns]
        df = df[keep_cols].copy()

        df["N"] = pd.to_numeric(df["N_meta"], errors="coerce")
        df = df.dropna(subset=["N", "filename", "model"]).copy()
        df["N"] = df["N"].astype(int)

        # 布尔化：ground_truth / pred_llm_yes
        df["ground_truth_bool"] = to_nullable_bool(df["ground_truth"])
        df["packing_pred_bool"] = to_nullable_bool(df["pred_llm_yes"])

        if "pred_assignment_verified" in df.columns:
            df["packing_assignment_verified_bool"] = to_nullable_bool(df["pred_assignment_verified"])

        # 文件名标准化（用于和 VC 总表对齐）
        df["filename_norm"] = df["filename"].map(canonicalize_filename_for_match)

        # 去重：同一 (model, N, filename_norm) 只保留最后一条
        df = df.sort_index().drop_duplicates(subset=["model", "N", "filename_norm"], keep="last")

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"在目录中没有成功读取到 PerSample: {intermediate_dir}")

    out = pd.concat(all_rows, ignore_index=True)
    return out


# =========================
# 读取 CNF 侧预测（来自 VC 对齐总表）
# =========================
def load_cnf_predictions_from_vc_merged(vc_merged_xlsx: str) -> pd.DataFrame:
    """
    从 all_file_all_model_all_N_vc_and_cnf_prediction.xlsx 读取：
      model, N, file_name, cnf_label_IS_SAT, cnf_prediction_IS_SAT
    """
    df = pd.read_excel(vc_merged_xlsx)

    required = ["model", "N", "file_name", "cnf_label_IS_SAT", "cnf_prediction_IS_SAT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"VC 总表缺少必要列: {missing}")

    out = df[required].copy()
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out = out.dropna(subset=["N", "file_name", "model"]).copy()
    out["N"] = out["N"].astype(int)

    out["filename_norm"] = out["file_name"].map(canonicalize_filename_for_match)
    out["cnf_label_bool"] = to_nullable_bool(out["cnf_label_IS_SAT"])
    out["cnf_pred_bool"] = to_nullable_bool(out["cnf_prediction_IS_SAT"])

    # 去重
    out = out.sort_index().drop_duplicates(subset=["model", "N", "filename_norm"], keep="last")

    return out


# =========================
# 从 overall DataFrame / Excel 提取 summary（通用）
# =========================
def _safe_div_plain(num, den):
    return (num / den) if den else 0.0


def summarize_agree_disagree_from_overall_df(
    overall_df: pd.DataFrame,
    counterpart_name: str = "packing",  # "packing" 或 "vc"
) -> dict:
    """
    从 overall DataFrame（你代码里返回的 overall）提取 summary。
    overall_df 应该是只有一行的 DataFrame。
    """
    if overall_df is None or overall_df.empty:
        raise ValueError("overall_df is empty.")

    row = overall_df.iloc[0]

    # 兼容 object/float/int
    valid_rows_count = int(row["valid_rows_count"])
    agree_count = int(row["agree_count"])
    disagree_count = int(row["disagree_count"])
    agree_match_label_count = int(row["agree_match_label_count"])
    agree_mismatch_label_count = int(row["agree_mismatch_label_count"])
    cnf_correct_when_disagree_count = int(row["cnf_correct_when_disagree_count"])

    # 动态 counterpart（packing / vc）
    counterpart_count_col = f"{counterpart_name}_correct_when_disagree_count"
    if counterpart_count_col not in overall_df.columns:
        # 如果你传的是 packing，但列名实际是 vc，可自动兜底
        if "vc_correct_when_disagree_count" in overall_df.columns:
            counterpart_name = "vc"
            counterpart_count_col = "vc_correct_when_disagree_count"
        elif "packing_correct_when_disagree_count" in overall_df.columns:
            counterpart_name = "packing"
            counterpart_count_col = "packing_correct_when_disagree_count"
        else:
            raise KeyError(
                "Cannot find counterpart count column. "
                "Expected one of: vc_correct_when_disagree_count / packing_correct_when_disagree_count"
            )

    counterpart_correct_when_disagree_count = int(row[counterpart_count_col])

    # 比例
    agree_ratio_overall = _safe_div_plain(agree_count, valid_rows_count)
    disagree_ratio_overall = _safe_div_plain(disagree_count, valid_rows_count)

    agree_match_label_ratio = _safe_div_plain(agree_match_label_count, agree_count)
    agree_mismatch_label_ratio = _safe_div_plain(agree_mismatch_label_count, agree_count)

    cnf_correct_when_disagree_ratio = _safe_div_plain(cnf_correct_when_disagree_count, disagree_count)
    counterpart_correct_when_disagree_ratio = _safe_div_plain(counterpart_correct_when_disagree_count, disagree_count)

    summary = {
        "counterpart_name": counterpart_name,
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
        }
    }
    return summary


def summarize_agree_disagree_from_xlsx(
    xlsx_path: str,
    overall_sheet: str = "overall",
    counterpart_name: str = "packing",
) -> dict:
    """
    从导出的 xlsx 的 overall sheet 读取 summary（如果你想后处理单独跑）
    """
    overall_df = pd.read_excel(xlsx_path, sheet_name=overall_sheet)
    return summarize_agree_disagree_from_overall_df(overall_df, counterpart_name=counterpart_name)


def format_agree_disagree_summary_text(summary: dict) -> str:
    """
    生成英文 summary 文本（避免嵌套 f-string 报错版本）
    """
    c = summary["counts"]
    p = summary["percents"]
    counterpart_name = summary["counterpart_name"]
    counterpart_upper = counterpart_name.upper()

    counterpart_pct_key = f"{counterpart_name}_correct_when_disagree_ratio_pct"
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
        f"{counterpart_upper} predictions match the label in "
        f"{counterpart_pct:.2f}% of instances."
    )
    return text


def print_agree_disagree_summary(summary: dict):
    """
    以更直观方式打印 summary（中英混合）
    """
    c = summary["counts"]
    p = summary["percents"]
    counterpart = summary["counterpart_name"]

    print("\n===== Overall Summary (CNF vs {}) =====".format(counterpart.upper()))
    print(f"valid_rows_count = {c['valid_rows_count']}")
    print(f"agree_count = {c['agree_count']} ({p['agree_ratio_overall_pct']:.2f}%)")
    print(f"disagree_count = {c['disagree_count']} ({p['disagree_ratio_overall_pct']:.2f}%)")

    print("\n[A] Agree subset (CNF and counterpart predict the same)")
    print(f"agree_match_label_count = {c['agree_match_label_count']} ({p['agree_match_label_ratio_pct']:.2f}%)")
    print(f"agree_mismatch_label_count = {c['agree_mismatch_label_count']} ({p['agree_mismatch_label_ratio_pct']:.2f}%)")

    counterpart_count_key = f"{counterpart}_correct_when_disagree_count"
    counterpart_pct_key = f"{counterpart}_correct_when_disagree_ratio_pct"

    print("\n[B] Disagree subset (CNF and counterpart predict differently)")
    print(f"cnf_correct_when_disagree_count = {c['cnf_correct_when_disagree_count']} ({p['cnf_correct_when_disagree_ratio_pct']:.2f}%)")
    print(f"{counterpart}_correct_when_disagree_count = {c[counterpart_count_key]} ({p[counterpart_pct_key]:.2f}%)")

    print("\n[English summary]")
    print(format_agree_disagree_summary_text(summary))


# =========================
# 计算 agree/disagree + label 指标（CNF vs 3D-packing）
# =========================
def compute_cnf_vs_3dpacking_agree_disagree_metrics(
    intermediate_dir: str,
    vc_merged_xlsx: str,
    output_dir: str,
    use_packing_pred_col: str = "packing_pred_bool",  # 你也可改成 packing_assignment_verified_bool 做更严格版本
):
    # 1) 读取 3D-packing PerSample
    pack_df = load_3dpacking_persample_from_intermediate(intermediate_dir)

    # 2) 读取 CNF 预测（来自 VC 对齐总表）
    cnf_df = load_cnf_predictions_from_vc_merged(vc_merged_xlsx)

    # 3) 合并（按 model, N, filename）
    merged = pd.merge(
        pack_df,
        cnf_df[["model", "N", "filename_norm", "cnf_label_bool", "cnf_pred_bool"]],
        on=["model", "N", "filename_norm"],
        how="inner",
        suffixes=("", "_cnf")
    )

    if merged.empty:
        raise RuntimeError("合并后为空：请检查 model 名称、N 值、filename(.txt/.cnf) 匹配情况。")

    # 4) 准备 label：优先用 CNF label（来自 VC 总表），也可与 3D ground_truth 做一致性检查
    merged["label_bool"] = merged["cnf_label_bool"]

    # 可选一致性检查（不强制）
    if "ground_truth_bool" in merged.columns:
        both_notna = merged["ground_truth_bool"].notna() & merged["label_bool"].notna()
        if both_notna.any():
            mismatch_cnt = int((merged.loc[both_notna, "ground_truth_bool"] != merged.loc[both_notna, "label_bool"]).sum())
            if mismatch_cnt > 0:
                print(f"[WARN] 3D ground_truth 与 CNF label 不一致的条目数: {mismatch_cnt}")

    # 5) 选择 3D-packing 的预测列（默认 pred_llm_yes）
    if use_packing_pred_col not in merged.columns:
        raise ValueError(f"use_packing_pred_col 不存在: {use_packing_pred_col}")

    merged["packing_pred"] = merged[use_packing_pred_col]

    # 仅保留可比较行
    valid = merged[
        merged["label_bool"].notna()
        & merged["cnf_pred_bool"].notna()
        & merged["packing_pred"].notna()
    ].copy()

    # 7) 模型白名单过滤（只统计图中出现的模型）
    before_rows = len(valid)
    valid = valid[valid["model"].astype(str).str.strip().isin(MODEL_WHITELIST)].copy()
    after_rows = len(valid)

    print(f"[INFO] 模型白名单过滤: {before_rows} -> {after_rows}")
    print(f"[INFO] 保留模型: {sorted(valid['model'].astype(str).unique().tolist())}")

    if valid.empty:
        raise RuntimeError("白名单过滤后 valid 为空，请检查模型名称是否一致（大小写/空格/命名差异）。")
    # 6) 关系列
    valid["cnf_pack_agree"] = (valid["cnf_pred_bool"] == valid["packing_pred"]).astype("boolean")
    valid["cnf_pack_disagree"] = (valid["cnf_pred_bool"] != valid["packing_pred"]).astype("boolean")

    valid["cnf_correct_vs_label"] = (valid["cnf_pred_bool"] == valid["label_bool"]).astype("boolean")
    valid["packing_correct_vs_label"] = (valid["packing_pred"] == valid["label_bool"]).astype("boolean")

    def summarize_group(g: pd.DataFrame) -> pd.Series:
        total_valid = int(len(g))

        agree_mask = (g["cnf_pack_agree"] == True)
        disagree_mask = (g["cnf_pack_disagree"] == True)

        agree_count = int(agree_mask.sum())
        disagree_count = int(disagree_mask.sum())

        # A. agree 子集：CNF=3D-packing 时，看其与 label 是否一致
        agree_match_label_count = _sum_bool(g.loc[agree_mask, "cnf_correct_vs_label"])  # agree 时看 cnf 或 packing 都一样
        agree_mismatch_label_count = agree_count - agree_match_label_count

        # B. disagree 子集：分别看 CNF 和 3D-packing 谁更接近 label
        cnf_correct_when_disagree_count = _sum_bool(g.loc[disagree_mask, "cnf_correct_vs_label"])
        packing_correct_when_disagree_count = _sum_bool(g.loc[disagree_mask, "packing_correct_vs_label"])

        out = {
            "valid_rows_count": total_valid,

            # ===== A. agree 子集 =====
            "agree_count": agree_count,
            "agree_match_label_count": agree_match_label_count,
            "agree_mismatch_label_count": agree_mismatch_label_count,
            "agree_match_label_ratio": _safe_ratio(agree_match_label_count, agree_count),
            "agree_mismatch_label_ratio": _safe_ratio(agree_mismatch_label_count, agree_count),

            # ===== B. disagree 子集 =====
            "disagree_count": disagree_count,
            "cnf_correct_when_disagree_count": cnf_correct_when_disagree_count,
            "cnf_correct_when_disagree_ratio": _safe_ratio(cnf_correct_when_disagree_count, disagree_count),
            "packing_correct_when_disagree_count": packing_correct_when_disagree_count,
            "packing_correct_when_disagree_ratio": _safe_ratio(packing_correct_when_disagree_count, disagree_count),
        }
        return pd.Series(out)

    summary_by_model_N = (
        valid.groupby(["model", "N"], dropna=False)
        .apply(summarize_group)
        .reset_index()
        .sort_values(["model", "N"], kind="mergesort")
        .reset_index(drop=True)
    )

    overall = summarize_group(valid).to_frame().T
    overall.insert(0, "scope", "ALL")

    # 导出
    os.makedirs(output_dir, exist_ok=True)
    out_xlsx = os.path.join(output_dir, "cnf_vs_3dpacking_agree_disagree_with_label_metrics.xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_by_model_N.to_excel(writer, sheet_name="by_model_N", index=False)
        overall.to_excel(writer, sheet_name="overall", index=False)
        valid.to_excel(writer, sheet_name="merged_valid_details", index=False)

    print(f"[OK] 已保存: {out_xlsx}")
    print(f"[INFO] 合并有效样本数: {len(valid)}")
    # === 新增：直接从 overall DataFrame 生成 summary ===
    overall_summary = summarize_agree_disagree_from_overall_df(overall, counterpart_name="packing")
    print_agree_disagree_summary(overall_summary)

    # 如需把 summary 也返回出去：
    return summary_by_model_N, overall, valid, overall_summary


if __name__ == "__main__":
    # 3D-packing 的中间 PerSample 文件目录（你给的路径）
    intermediate_dir = (
        "/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/"
        "cnf_to_3D_packing/analysis/three_ways_evaluation/intermediate_results"
    )

    # VC 对齐总表（提供 CNF prediction + CNF label）
    vc_merged_xlsx = (
        "/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/analysis/"
        "cnf_and_vertex_cover_equivalence/all_file_all_model_all_N_vc_and_cnf_prediction.xlsx"
    )

    # 输出目录（你可以放在 3D-packing 的 analysis/three_ways_evaluation 下）
    output_dir = (
        "/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_3D_packing/"
        "cnf_to_3D_packing/analysis/three_ways_evaluation"
    )

    # 默认使用 3D-packing 的 pred_llm_yes 作为 3D-packing 预测
    by_model_N, overall, valid, overall_summary = compute_cnf_vs_3dpacking_agree_disagree_metrics(
        intermediate_dir=intermediate_dir,
        vc_merged_xlsx=vc_merged_xlsx,
        output_dir=output_dir,
        use_packing_pred_col="packing_pred_bool",
    )
    print("\n=== by_model_N ===")
    print(by_model_N.head(50))
    print("\n=== overall ===")
    print(overall)

    print("\n=== overall_summary (dict) ===")
    print(overall_summary)

    print("\n=== formatted summary text ===")
    print(format_agree_disagree_summary_text(overall_summary))

