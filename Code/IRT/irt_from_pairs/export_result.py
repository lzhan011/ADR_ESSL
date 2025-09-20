import pandas as pd
from irt_core import extract_best_params_from_current_pyirt
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


# ===== Publication-grade styling =====
from matplotlib import rcParams

def set_pub_style(
    base=20,          # main font size (axes labels)
    small=18,         # tick labels & legend
    tiny=16,          # annotations etc.
    title=22,         # axes title size
    figtitle=24,      # figure suptitle (if any)
    linewidth=1.6,    # lines if any
    markersize=60,    # default scatter size "s"
    grid_alpha=0.28
):
    """
    Set large, consistent fonts & sizes so figures remain legible after scaling.
    Tune 'base' upward if your final PDF shrinks figures further.
    """
    rcParams.update({
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": title,
        "figure.titlesize": figtitle,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": base,
        "lines.linewidth": linewidth,
        # Optional: make saved figures tight by default
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    # Return sizes to use in functions (for annotations & markers)
    return {
        "BASE": base, "SMALL": small, "TINY": tiny,
        "TITLE": title, "FIGTITLE": figtitle,
        "MARKER_S": markersize, "GRID_ALPHA": grid_alpha
    }

# Call once before plotting (also safe to call again with different sizes)
PUB = set_pub_style(base=20, small=18, tiny=16, title=22, figtitle=24)


def export_res(base, pair_or_instance_level):
    # 快速看一下结构，确认不再是 dict 取索引报错
    import json
    with open(base / "pyirt_2pl/best_parameters.json", "r") as f:
        P = json.load(f)
    print(type(P.get("item_ids")), list(P.get("item_ids", {}))[:5])  # 如果原来是 dict，这里会显示 dict

    for tag in ("2pl", "3pl", "4pl"):
        best = base / f"pyirt_{tag}" / "best_parameters.json"
        abil_csv = base / f"abilities_{tag}.csv"
        items_csv = base / f"item_params_{tag}.csv"
        abil, items = extract_best_params_from_current_pyirt(str(best), pair_or_instance_level=pair_or_instance_level)
        abil.to_csv(abil_csv, index=False)
        items.to_csv(items_csv, index=False)
        print(f"{tag}: {len(abil)} abilities, {len(items)} items ->",
              abil_csv.name, items_csv.name)


def show_and_plot(model_type):
    abil_csv = os.path.join(OUT_DIR, f"abilities_{model_type}.csv")
    item_csv = os.path.join(OUT_DIR, f"item_params_{model_type}.csv")

    if not os.path.exists(abil_csv) or not os.path.exists(item_csv):
        print(f"[skip] {model_type.upper()} 没有结果")
        return

    # 读取能力
    abil = pd.read_csv(abil_csv)
    print(f"\n=== {model_type.upper()} abilities 排序 ===")
    print(abil.sort_values("theta", ascending=False).to_string(index=False))

    # 读取 item 参数
    items = pd.read_csv(item_csv)

    # 兼容列名
    if "a" not in items.columns:
        for alt in ("alpha", "discrimination"):
            if alt in items.columns:
                items["a"] = items[alt]
                break
    if "b" not in items.columns:
        for alt in ("beta", "difficulty"):
            if alt in items.columns:
                items["b"] = items[alt]
                break

    # 画散点图
    plt.figure(figsize=(6.5, 5.4))  # a bit wider helps text
    plt.scatter(items["b"], items["a"], alpha=0.75, s=PUB["MARKER_S"])
    plt.xlabel("Difficulty (b)")
    plt.ylabel("Discrimination (a)")
    plt.title(f"{model_type.upper()} Item Parameters (a vs b)")
    plt.grid(True, linestyle="--", alpha=PUB["GRID_ALPHA"])
    ax = plt.gca()
    ax.tick_params(labelsize=rcParams["xtick.labelsize"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"items_ab_{model_type}.png"))
    plt.close()
    print(f"[图保存] items_ab_{model_type}.png")

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_irt_abilities(out_dir: str,
                          files: dict = None,
                          out_csv: str = "abilities_comparison.csv",
                          out_png: str = "abilities_comparison.png"):
    """
    比较 2PL/3PL/4PL 三种 IRT 模型下的能力估计，并生成对比表 + 柱状图。

    参数:
    -------
    out_dir : str
        输出目录，保存图表。
    files : dict
        {'2pl': 'abilities_2pl.csv',
         '3pl': 'abilities_3pl.csv',
         '4pl': 'abilities_4pl.csv'} (路径均在 out_dir 下)
    out_csv : str
        输出的合并 CSV 文件名。
    out_png : str
        输出的柱状图 PNG 文件名。

    返回:
    -------
    pd.DataFrame
        合并后的宽表 (model, theta_2pl, theta_3pl, theta_4pl)
    """
    if files is None:
        files = {
            "2pl": os.path.join(out_dir, "abilities_2pl.csv"),
            "3pl": os.path.join(out_dir, "abilities_3pl.csv"),
            "4pl": os.path.join(out_dir, "abilities_4pl.csv"),
        }

    dfs = {}
    for tag, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少文件: {path}")
        df = pd.read_csv(path)
        if "theta" not in df.columns:
            raise KeyError(f"{path} 缺少 'theta' 列")
        dfs[tag] = df.rename(columns={"theta": f"theta_{tag}"})

    # 按 model 合并
    merged = dfs["2pl"]
    for tag in ["3pl", "4pl"]:
        merged = pd.merge(merged, dfs[tag], on="model", how="outer")

    # 排序：按 2PL theta 从大到小
    merged = merged.sort_values(by="theta_2pl", ascending=False).reset_index(drop=True)

    # 保存合并表
    merged_csv = os.path.join(out_dir, out_csv)
    merged.to_csv(merged_csv, index=False)

    # 绘制分组柱状图
    x = np.arange(len(merged["model"]))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.bar(x - width, merged["theta_2pl"], width, label="2PL")
    ax.bar(x, merged["theta_3pl"], width, label="3PL")
    ax.bar(x + width, merged["theta_4pl"], width, label="4PL")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["model"], rotation=45, ha="right")
    ax.set_ylabel("Theta (Ability)")
    ax.set_title("IRT Abilities Comparison (2PL vs 3PL vs 4PL)")
    leg = ax.legend()
    # ensure legend follows our sizes even on older MPL
    try:
        leg.set_title(leg.get_title().get_text(), prop={"size": rcParams["legend.title_fontsize"]})
        for txt in leg.get_texts():
            txt.set_fontsize(rcParams["legend.fontsize"])
    except Exception:
        pass
    ax.tick_params(labelsize=rcParams["xtick.labelsize"])
    plt.tight_layout()

    merged_png = os.path.join(out_dir, out_png)
    plt.savefig(merged_png)
    plt.close()

    print(f"[保存成功] {merged_csv}")
    print(f"[保存成功] {merged_png}")

    return merged


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def plot_irt_abilities_with_scatter(
        abilities_dict: dict,
        out_dir: str = ".",
        prefix: str = "irt_compare",
        add_spearman: bool = False,
        annotate_points: bool = True
):
    """
    生成 IRT 能力比较图：柱状图 + 散点图（含皮尔逊 r、可选 Spearman ρ、拟合线）。
    """
    os.makedirs(out_dir, exist_ok=True)

    # —— 合并：每列是不同模型类型的 theta ——
    merged = None
    for tag, df in abilities_dict.items():
        df = df[['model', 'theta']].copy()
        df = df.rename(columns={'theta': f'{tag}_theta'})
        merged = df if merged is None else merged.merge(df, on='model', how='outer')

    # 按 2PL 排序展示（若有）
    sort_key = '2PL_theta' if '2PL_theta' in merged.columns else merged.columns[1]
    sort_key = '4PL_theta' if '4PL_theta' in merged.columns else (
        '2PL_theta' if '2PL_theta' in merged.columns else merged.columns[1]
    )
    merged = merged.sort_values(sort_key, ascending=False).reset_index(drop=True)

    # —— 柱状图 ——
    cols_to_plot = [c for c in ('2PL_theta', '3PL_theta', '4PL_theta') if c in merged.columns]
    ax = merged.set_index('model')[cols_to_plot].plot(kind='bar', figsize=(14, 6))
    plt.ylabel("Theta (Ability)")
    plt.title("IRT Abilities Comparison (2PL vs 3PL vs 4PL)")
    plt.xticks(rotation=45, ha="right")
    ax.tick_params(labelsize=rcParams["xtick.labelsize"])
    ax.legend()
    plt.tight_layout()
    # >>> 关键：保存并定义 bar_path
    bar_path = os.path.join(out_dir, f"{prefix}_bar.png")
    plt.savefig(bar_path, dpi=160)
    plt.close()

    # —— 散点图：相关、拟合线、标注 ——
    def _scatter_with_stats(x, y, xlab, ylab, fname):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        plt.figure(figsize=(6.6, 6.4))
        plt.scatter(x, y, s=PUB["MARKER_S"])

        # 回归拟合线（最小二乘）
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            plt.plot(xx, m * xx + b, linewidth=1)

        # 相关
        if len(x) >= 2:
            r, p = pearsonr(x, y)
            title = f"Ability Correlation: {xlab} vs {ylab}\nPearson r={r:.3f}, p={p:.2g}"
            if add_spearman:
                rho, p2 = spearmanr(x, y)
                title += f" | Spearman ρ={rho:.3f}, p={p2:.2g}"
        else:
            title = f"Ability Correlation: {xlab} vs {ylab}\n(n<2)"

        plt.title(title)
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.axhline(0, linewidth=0.6); plt.axvline(0, linewidth=0.6)

        if annotate_points:
            # 用更大的注释字号
            for i, row in merged[mask].iterrows():
                plt.text(row[xlab], row[ylab], row['model'], fontsize=PUB["TINY"])

        plt.tight_layout()
        outp = os.path.join(out_dir, fname)
        plt.savefig(outp, dpi=160)
        plt.close()
        return outp

    scatter_paths = []
    pairs = []
    if '2PL_theta' in merged.columns and '3PL_theta' in merged.columns:
        pairs.append(('2PL_theta', '3PL_theta'))
    if '2PL_theta' in merged.columns and '4PL_theta' in merged.columns:
        pairs.append(('2PL_theta', '4PL_theta'))
    if '3PL_theta' in merged.columns and '4PL_theta' in merged.columns:
        pairs.append(('3PL_theta', '4PL_theta'))

    for xcol, ycol in pairs:
        spath = _scatter_with_stats(
            merged[xcol].to_numpy(dtype=float),
            merged[ycol].to_numpy(dtype=float),
            xcol, ycol,
            f"{prefix}_scatter_{xcol}_vs_{ycol}.png"
        )
        scatter_paths.append(spath)

    return {"bar": bar_path, "scatter": scatter_paths, "merged": merged}



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# —— 若你已有这个函数，可复用；否则保留这段 ——
def _get_param_cols(item_df: pd.DataFrame):
    name_map = {"a": None, "b": None, "c": None, "d": None}
    candidates = {
        "a": ["a", "discrimination", "alpha"],
        "b": ["b", "difficulty", "beta"],
        "c": ["c", "guessing", "lower_asymptote"],
        "d": ["d", "upper_asymptote", "u"],
    }
    for k, opts in candidates.items():
        for c in opts:
            if c in item_df.columns:
                name_map[k] = c
                break
    return name_map

# —— 复用你之前的构造 item_id 的逻辑 ——
def make_pair_uid_from_pairs_df(df_pairs: pd.DataFrame) -> pd.Series:
    if "pair_base_name" in df_pairs.columns:
        return df_pairs["pair_base_name"].astype(str)
    return df_pairs["original_file"].astype(str) + "||" + df_pairs["fixed_file"].astype(str)

def _build_item_N_map(all_pairs_path: str, level) -> pd.DataFrame:
    """
    从 all_pairs.xlsx 里抽取 item_id 与 N 的一一对应表（去重）。
    """
    df_pairs = pd.read_excel(all_pairs_path)

    if level.lower() == "instance":
        need = ["original_file", "fixed_file", "N"]
        missing = [c for c in need if c not in df_pairs.columns]
        if missing:
            raise KeyError(f"all_pairs 缺少列: {missing}")
        # 每个 pair 拆成两个 instance（文件）并继承同一个 N
        left = df_pairs[["original_file", "N"]].rename(columns={"original_file": "item_id"})
        right = df_pairs[["fixed_file", "N"]].rename(columns={"fixed_file": "item_id"})
        itemN = pd.concat([left, right], ignore_index=True).dropna(subset=["item_id"])
        itemN["item_id"] = itemN["item_id"].astype(str)
        itemN = itemN.drop_duplicates()
        return itemN

    if "item_id" not in df_pairs.columns:
        df_pairs = df_pairs.copy()
        df_pairs["item_id"] = make_pair_uid_from_pairs_df(df_pairs)
    if "N" not in df_pairs.columns:
        raise KeyError("all_pairs 缺少列 N。")

    itemN = df_pairs[["item_id", "N"]].drop_duplicates()
    return itemN

def plot_items_ab_byN(items_csv: str,
                      all_pairs_path: str,
                      model_tag: str,
                      out_png: str,
                      max_legend_entries: int = 15,
                      pair_or_instance_level="pair"):
    """
    读取 item 参数 (items_csv: 含 item_id, a,b[,c,d] 等)，
    合并 all_pairs.xlsx 提供的 N，按 N 着色绘制 b–a 散点图。
    颜色固定映射：
      5→绿, 8→蓝, 10→黄, 25→紫, 50→褐, 60→红；其它值为灰色。
    """
    items = pd.read_csv(items_csv)
    if "item_id" not in items.columns:
        raise KeyError(f"{items_csv} 缺少 item_id 列。")

    # —— 找 a/b 列 —— #
    name_map = _get_param_cols(items)
    a_col, b_col = name_map["a"], name_map["b"]
    if a_col is None or b_col is None:
        raise ValueError(f"在 {items_csv} 找不到 a/b 列，请检查列名：{list(items.columns)}")

    # —— 合并 all_pairs 的 N —— #
    itemN = _build_item_N_map(all_pairs_path, level=pair_or_instance_level)
    df = items.merge(itemN, on="item_id", how="left").copy()
    df[a_col] = pd.to_numeric(df[a_col], errors="coerce")
    df[b_col] = pd.to_numeric(df[b_col], errors="coerce")

    # 分组键（字符串），便于做 legend 标签
    df["N_group"] = df["N"].astype("Int64").astype(str)
    df.loc[df["N"].isna(), "N_group"] = "NA"

    # —— 固定颜色映射（按你的要求）—— #
    # 使用 Matplotlib 标准颜色名：tab:green, tab:blue, gold, purple, saddlebrown, tab:red
    fixed_color_map = {
        "5":  "tab:green",     # 绿色
        "8":  "tab:blue",      # 蓝色
        "10": "gold",          # 黄色
        "25": "purple",        # 紫色
        "50": "saddlebrown",   # 褐色
        "60": "tab:red",       # 红色
    }
    default_color = "lightgray"    # 其它 N 的颜色

    # —— 图例顺序：按数值从小到大；NA（缺失）和其它放在后面 —— #
    # 先拿到所有组，再拆成三类：固定映射里的、有数字但不在映射里的、NA
    groups_all = sorted(df["N_group"].unique(), key=lambda x: (x == "NA", x))
    fixed_order = ["5", "8", "10", "25", "50", "60"]  # 指定的顺序
    fixed_present = [g for g in fixed_order if g in groups_all]
    others_numeric = sorted(
        [g for g in groups_all if g not in fixed_present and g not in ("NA",)],
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 1_000_000)
    )
    groups = fixed_present + others_numeric + (["NA"] if "NA" in groups_all else [])

    # 若组别太多，可按需压缩到 “others”（保留你的 max_legend_entries 逻辑）
    if len(groups) > max_legend_entries:
        keep = groups[:max_legend_entries - 1]
        df["N_group"] = df["N_group"].where(df["N_group"].isin(keep), other="others")
        # 重新确定最终 groups（保持 keep 顺序，others 放最后）
        groups = [g for g in keep if g in df["N_group"].unique()]
        if "others" in df["N_group"].unique():
            groups.append("others")

    # —— 绘图 —— #
    plt.figure(figsize=(7.8, 5.6))
    for g in groups:
        sub = df[df["N_group"] == g]
        if sub.empty:
            continue
        color = fixed_color_map.get(g, default_color)
        label = f"N={g}" if g not in ("NA", "others") else g
        plt.scatter(sub[b_col], sub[a_col], s=PUB["MARKER_S"], alpha=0.85, label=label, c=color)

    plt.xlabel("Item difficulty (b)")
    plt.ylabel("Item discrimination (a)")
    plt.title(f"Item map by N — {model_tag.upper()}")
    plt.grid(True, linestyle="--", alpha=PUB["GRID_ALPHA"])


    # 图例放外侧，按上面 groups 顺序
    leg = plt.legend(
        title="N", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0
    )
    # 兼容不同 Matplotlib 版本，调大图例点的大小
    try:
        leg.set_title(leg.get_title().get_text(), prop={"size": rcParams["legend.title_fontsize"]})
        for txt in leg.get_texts():
            txt.set_fontsize(rcParams["legend.fontsize"])
    except Exception:
        pass
    ax = plt.gca()
    ax.tick_params(labelsize=rcParams["xtick.labelsize"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()





def plot_all_items_byN(out_dir: str, all_pairs_path: str, pair_or_instance_level):
    """
    批量：对 2pl/3pl/4pl 的 item 参数做“按 N 上色”的 a-b 散点图。

    输入：
      out_dir 下应存在 item_params_2pl.csv / item_params_3pl.csv / item_params_4pl.csv
      all_pairs_path: pair_predictions_all_models.xlsx (包含 N 和 item_id 对应关系)

    输出：
      items_ab_2pl_byN.png / items_ab_3pl_byN.png / items_ab_4pl_byN.png
    """
    plan = [
        ("2pl", os.path.join(out_dir, "item_params_2pl.csv"),
         os.path.join(out_dir, "items_ab_2pl_byN.png")),
        ("3pl", os.path.join(out_dir, "item_params_3pl.csv"),
         os.path.join(out_dir, "items_ab_3pl_byN.png")),
        ("4pl", os.path.join(out_dir, "item_params_4pl.csv"),
         os.path.join(out_dir, "items_ab_4pl_byN.png")),
    ]

    for tag, items_csv, out_png in plan:
        if os.path.exists(items_csv):
            try:
                plot_items_ab_byN(items_csv, all_pairs_path, tag, out_png, pair_or_instance_level=pair_or_instance_level)
                print(f"[图保存] {out_png}")
            except Exception as e:
                print(f"[warn] 绘制 {tag} 失败：{e}")
        else:
            print(f"[skip] 未找到 {items_csv}")



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_item_param_histograms(
    out_dir: str,
    files,
    params: tuple = ("a", "b", "c", "d"),
    nbins: int = 40,
    density: bool = True,
    style: str = "overlay",          # "overlay" | "facet"
    clip_a_by_quantiles: bool = True,# 避免 3PL 极端 a 撑爆横轴
    clip_q: tuple = (0.01, 0.99),    # 截尾分位点
    figsize_overlay: tuple = (8, 5),
    figsize_facet: tuple = (12, 8),
    dpi: int = 180,
    alpha: float = 0.45
):
    """
    读取 2PL/3PL/4PL 的 item 参数 csv，绘制所选参数的直方图对比。

    Parameters
    ----------
    out_dir : str
        输出目录（也是 csv 所在目录）。图会保存在这里。
    files : dict | None
        若为 None，默认在 out_dir 下寻找：
        {"2PL": "item_params_2pl.csv", "3PL": "item_params_3pl.csv", "4PL": "item_params_4pl.csv"}
        也可自定义完整路径。
    params : tuple
        要绘制的参数集合，取自 {"a","b","c","d"}，按顺序生成。
    nbins : int
        直方图 bin 数。
    density : bool
        是否标准化为密度。
    style : str
        "overlay"：三模型叠加一张图；"facet"：每模型一张小图组成网格。
    clip_a_by_quantiles : bool
        对参数 a 进行温和截尾（例如 1%~99%），避免极端值影响可读性。
    clip_q : tuple
        截尾分位点。
    figsize_overlay / figsize_facet : tuple
        图尺寸。
    dpi : int
        输出分辨率。
    alpha : float
        叠加直方图透明度。

    Returns
    -------
    dict
        { "<param>": <path or list-of-paths> }
    """
    os.makedirs(out_dir, exist_ok=True)
    # --- 1) 确定 csv 路径 ---
    if files is None:
        files = {
            "2PL": os.path.join(out_dir, "item_params_2pl.csv"),
            "3PL": os.path.join(out_dir, "item_params_3pl.csv"),
            "4PL": os.path.join(out_dir, "item_params_4pl.csv"),
        }

    # --- 2) 读取并标准化列名 ---
    def _detect_cols(df: pd.DataFrame):
        cand = {
            "a": ["a", "alpha", "discrimination"],
            "b": ["b", "beta", "difficulty"],
            "c": ["c", "guessing", "lower_asymptote"],
            "d": ["d", "upper_asymptote", "u"],
        }
        res = {}
        for k, opts in cand.items():
            res[k] = next((c for c in opts if c in df.columns), None)
        return res

    def _load_params(tag: str, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{tag}] file not found: {path}")
        df = pd.read_csv(path)
        cols = _detect_cols(df)
        # 只重命名找到的列
        rename_map = {v: k for k, v in cols.items() if v is not None and v != k}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    models = {}
    for tag, p in files.items():
        try:
            df = _load_params(tag, p)
            models[tag] = df
        except Exception as e:
            print(f"[warn] {tag} failed to load: {e}")

    if not models:
        raise RuntimeError("No model csv loaded. Check 'files' paths.")

    # --- 3) 统一 bins（可对 a 做截尾后再取范围） ---
    def _common_bins(series_list, nbins=40, clip=None):
        vals = pd.concat([pd.Series(s).dropna() for s in series_list], ignore_index=True)
        if vals.empty:
            return None
        if clip is not None:
            lo, hi = np.nanquantile(vals, clip)
            vals = vals.clip(lo, hi)
        vmin, vmax = float(vals.min()), float(vals.max())
        if np.isclose(vmin, vmax):
            vmin -= 1e-6; vmax += 1e-6
        return np.linspace(vmin, vmax, nbins + 1)

    # --- 4) 绘图 ---
    out_paths = {}
    for param in params:
        # 收集该参数各模型数据
        data = {k: v[param].astype(float).dropna()
                for k, v in models.items() if param in v.columns}
        if not data:
            print(f"[skip] no '{param}' in any file")
            continue

        clip = clip_q if (param == "a" and clip_a_by_quantiles) else None
        bins = _common_bins(list(data.values()), nbins=nbins, clip=clip)
        if bins is None:
            print(f"[skip] '{param}' empty")
            continue

        if style == "overlay":
            plt.figure(figsize=figsize_overlay)
            for tag, s in data.items():
                vals = s.copy()
                if clip is not None and not vals.empty:
                    lo, hi = np.nanquantile(vals, clip)
                    vals = vals.clip(lo, hi)
                plt.hist(vals, bins=bins, alpha=alpha, label=tag, density=density)
            plt.title(f"Distribution of parameter '{param}' (2PL vs 3PL vs 4PL)")
            plt.xlabel(param);
            plt.ylabel("Density" if density else "Count")
            leg = plt.legend()
            try:
                leg.set_title(leg.get_title().get_text(), prop={"size": rcParams["legend.title_fontsize"]})
                for txt in leg.get_texts():
                    txt.set_fontsize(rcParams["legend.fontsize"])
            except Exception:
                pass
            ax = plt.gca()
            ax.tick_params(labelsize=rcParams["xtick.labelsize"])
            plt.tight_layout()
            out_png = os.path.join(out_dir, f"param_{param}_hist_overlay.png")
            plt.savefig(out_png, dpi=dpi); plt.close()
            out_paths[param] = out_png

        elif style == "facet":
            n = len(data)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize_facet, squeeze=False)
            axes = axes.flatten()
            for ax, (tag, s) in zip(axes, data.items()):
                vals = s.copy()
                if clip is not None and not vals.empty:
                    lo, hi = np.nanquantile(vals, clip)
                    vals = vals.clip(lo, hi)
                ax.hist(vals, bins=bins, alpha=0.8, density=density)
                ax.set_title(f"{tag} — '{param}'")
                ax.set_xlabel(param);
                ax.set_ylabel("Density" if density else "Count")
                ax.tick_params(labelsize=rcParams["xtick.labelsize"])
            # 清空多余子图
            for ax in axes[len(data):]:
                ax.axis("off")
            plt.tight_layout()
            out_png = os.path.join(out_dir, f"param_{param}_hist_facet.png")
            plt.savefig(out_png, dpi=dpi); plt.close()
            out_paths[param] = out_png
        else:
            raise ValueError("style must be 'overlay' or 'facet'")

        print(f"[saved] {param}: {out_paths[param]}")

    return out_paths

if __name__ == '__main__':
    # export_res()
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    PUB = set_pub_style(base=20, small=18, tiny=16, title=22, figtitle=24)

    # 你的输出目录
    OUT_DIR = "/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result/analysis/irt_outputs"
    pair_or_instance_level = 'pair'  #pair  instance
    OUT_DIR = os.path.join(OUT_DIR, pair_or_instance_level)
    export_res(Path(OUT_DIR), pair_or_instance_level)
    for mt in ["2pl", "3pl", "4pl"]:
        show_and_plot(mt)

    df_comp = compare_irt_abilities(OUT_DIR)
    print(df_comp.head())

    abil2 = pd.read_csv(os.path.join(OUT_DIR, "abilities_2pl.csv"))
    abil3 = pd.read_csv(os.path.join(OUT_DIR, "abilities_3pl.csv"))
    abil4 = pd.read_csv(os.path.join(OUT_DIR, "abilities_4pl.csv"))

    res = plot_irt_abilities_with_scatter(
        {"2PL": abil2, "3PL": abil3, "4PL": abil4},
        out_dir=OUT_DIR,
        prefix="models_ability",
        add_spearman=True,  # 需要就开
        annotate_points=True
    )

    print("柱状图：", res["bar"])
    print("散点图：", res["scatter"])
    # 如需保存合并表：
    res["merged"].to_csv(os.path.join(OUT_DIR, "models_ability_merged.csv"), index=False)

    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    three_level_similar = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result'
    three_level_similar_analysis_dir = os.path.join(three_level_similar, 'analysis')
    four_level_similar_analysis_dir = os.path.join(three_level_similar_analysis_dir, 'three_ways_evaluation')

    # =====================
    # Edit these paths/params before running
    # =====================
    ALL_PAIRS_XLSX_PATH = os.path.join(four_level_similar_analysis_dir,
                                  "pair_predictions_all_models.xlsx")  # <-- 修改为你真实的 all_pairs 路径


    # 拟合与导出完以后：
    plot_all_items_byN(OUT_DIR, all_pairs_path=ALL_PAIRS_XLSX_PATH, pair_or_instance_level=pair_or_instance_level)

    # 1) 叠加样式：三模型同图对比（推荐投稿正文用）
    paths_overlay = plot_item_param_histograms(
        out_dir=OUT_DIR,
        files={
            "2PL": os.path.join(OUT_DIR, "item_params_2pl.csv"),
            "3PL": os.path.join(OUT_DIR, "item_params_3pl.csv"),
            "4PL": os.path.join(OUT_DIR, "item_params_4pl.csv"),
        },
        params=("a", "b"),  # 只画 a 与 b
        style="overlay",
        nbins=40,
        clip_a_by_quantiles=True
    )

    # 2) 分面样式：每模型单独子图（适合补充材料）
    paths_facet = plot_item_param_histograms(
        out_dir=OUT_DIR,
        files={
            "2PL": os.path.join(OUT_DIR, "item_params_2pl.csv"),
            "3PL": os.path.join(OUT_DIR, "item_params_3pl.csv"),
            "4PL": os.path.join(OUT_DIR, "item_params_4pl.csv"),
        },
        params=("a", "b", "c", "d"),
        style="facet",
        nbins=40,
        clip_a_by_quantiles=True
    )



