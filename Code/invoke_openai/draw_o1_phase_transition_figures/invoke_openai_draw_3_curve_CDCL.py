# -*- coding: utf-8 -*-
"""
从目录 /work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/cnf_results_openai_o1
读取每个文件中的 DIMACS CNF，用 Minisat22 求解：
- 判定 SAT/UNSAT
- 记录分支（decisions）
- 汇总每个 (N, alpha) 上的 mean/median branches & Prob(SAT)
- 把逐实例结果与聚合结果写入目标目录
- 为每个 N 画一张相变图（mean/median branches 与 Prob(SAT)）
"""

import os
import re
from typing import List, Tuple, Optional
from statistics import mean, median

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pysat.solvers import Minisat22


# ========= 配置路径 =========
INPUT_DIR = "/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/draw_o1_phase_transition_figures/draw_o1_cnf_alpha_3_6_N_75"
output_dir = "draw_o1_cnf_alpha_3_6_N_75_CDCL"  # 最终目录 = 该前缀 + 模型名

OUT_DIR    =  os.path.join(os.path.dirname(INPUT_DIR), output_dir)
FIG_DIR    = os.path.join(OUT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ========= 解析工具 =========
FNAME_RE = re.compile(
    r"""
    (?:^|_)N(?P<N>\d+)          # N<num>
    .*?
    (?:^|_)L(?P<L>\d+)          # L<num>
    .*?
    (?:^|_)alpha(?P<alpha>[\d.]+) # alpha<float>
    .*?
    (?:^|_)inst(?P<inst>\d+)    # inst<num>
    """,
    re.IGNORECASE | re.VERBOSE,
)

def parse_header_N_L(line: str) -> Optional[Tuple[int, int]]:
    """
    解析 'p cnf N L' 行，返回 (N, L) 或 None
    """
    parts = line.strip().split()
    if len(parts) >= 4 and parts[0].lower() == 'p' and parts[1].lower() == 'cnf':
        try:
            return int(parts[2]), int(parts[3])
        except Exception:
            return None
    return None


def read_dimacs(filepath: str) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    """
    读取 DIMACS 文件，返回 (clauses, N_from_header, L_from_header)
    - 忽略以 'c' 开头的注释行
    - 尽量容错，允许文件末尾有额外文本（例如模型输出）
    """
    clauses: List[List[int]] = []
    n_from_header = None
    l_from_header = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('c') or line.startswith('C'):
                # 注释
                continue
            if line.startswith('p') or line.startswith('P'):
                nl = parse_header_N_L(line)
                if nl:
                    n_from_header, l_from_header = nl
                continue

            # 尝试按 DIMACS 子句解析（以 0 结尾）
            # 允许行中有多余空白/制表符
            tokens = line.split()
            if tokens and tokens[-1] == '0':
                try:
                    lits = [int(tok) for tok in tokens[:-1]]
                except Exception:
                    # 该行不是标准子句（例如有非数字的噪声）；跳过
                    continue
                if lits:
                    clauses.append(lits)

    # 如果 header 未提供 L，则用子句数作为 L
    if l_from_header is None:
        l_from_header = len(clauses)

    return clauses, n_from_header, l_from_header


def find_k_n_alpha_from_name(fname: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int]]:
    """
    从文件名中提取 N, L, alpha, inst（容错）
    """
    m = FNAME_RE.search(fname)
    if not m:
        return None, None, None, None
    N     = int(m.group('N')) if m.group('N') else None
    L     = int(m.group('L')) if m.group('L') else None
    alpha = float(m.group('alpha')) if m.group('alpha') else None
    inst  = int(m.group('inst')) if m.group('inst') else None
    return N, L, alpha, inst


# ========= 主逻辑：遍历输入目录，逐文件求解 =========
results = []  # 收集逐实例结果（用于汇总/画图）

all_files = [
    f for f in sorted(os.listdir(INPUT_DIR))
    if os.path.isfile(os.path.join(INPUT_DIR, f))
]
print(f"Found {len(all_files)} files in {INPUT_DIR}")

for fname in tqdm(all_files, desc="Solving with Minisat22"):
    fpath = os.path.join(INPUT_DIR, fname)

    # 解析 CNF
    clauses, n_hdr, l_hdr = read_dimacs(fpath)
    if not clauses:
        # 空或无法解析，跳过
        continue

    # 从文件名提取 N/L/alpha/inst
    N_fn, L_fn, alpha_fn, inst_fn = find_k_n_alpha_from_name(fname)
    N = N_fn if N_fn is not None else n_hdr
    L = L_fn if L_fn is not None else l_hdr

    # 如果 alpha 缺失且 N 可用，则用 L/N 近似
    alpha = alpha_fn
    if alpha is None and (N is not None and N != 0 and L is not None):
        alpha = round(L / N, 6)

    # Minisat22 求解
    with Minisat22(bootstrap_with=clauses) as m:
        sat = m.solve()
        stats = m.accum_stats() or {}
        decisions = int(stats.get('decisions', 0))

    # 写逐实例结果到目标目录（与输入同名，*.cdcl.txt）
    out_txt = os.path.join(OUT_DIR, os.path.splitext(fname)[0] + ".txt")
    with open(out_txt, "w", encoding="utf-8") as fo:
        fo.write(f"c Source file: {fname}\n")
        if N is not None and L is not None:
            fo.write(f"p cnf {N} {L}\n")
        for cl in clauses:
            fo.write(" ".join(map(str, cl)) + " 0\n")
        fo.write("\n\n")
        fo.write(f"{'SATISFIABLE' if sat else 'UNSATISFIABLE'}\n")
        fo.write(f"brunches number: {decisions}\n")

    # 记录到内存
    results.append({
        "file": fname,
        "N": N,
        "L": L,
        "alpha": alpha,
        "inst": inst_fn,
        "sat": bool(sat),
        "decisions": decisions,
    })

# 保存逐实例表
if not results:
    raise SystemExit("No solvable DIMACS CNF found. Please check input directory or file formats.")

df = pd.DataFrame(results)
df.to_csv(os.path.join(FIG_DIR, "cdcl_instance_results.csv"), index=False)
print(f"Wrote per-instance results to {os.path.join(OUT_DIR, 'cdcl_instance_results.csv')}")

# 过滤缺失 alpha 或 N 的
df_clean = df.dropna(subset=["N", "alpha"]).copy()
df_clean["N"] = df_clean["N"].astype(int)
df_clean["alpha"] = df_clean["alpha"].astype(float)

# ========= 统计每个 (N, alpha) 的 mean/median branches 和 Prob(SAT) =========
agg = (
    df_clean
    .groupby(["N", "alpha"], as_index=False)
    .agg(
        mean_branches=("decisions", "mean"),
        median_branches=("decisions", "median"),
        prob_sat=("sat", "mean"),
        instances=("file", "count"),
    )
    .sort_values(["N", "alpha"])
)

# 保存聚合表
agg.to_csv(os.path.join(FIG_DIR, "cdcl_stats_by_N_alpha.csv"), index=False)
print(f"Wrote aggregate stats to {os.path.join(FIG_DIR, 'cdcl_stats_by_N_alpha.csv')}")

# ========= 为每个 N 画相变曲线 =========
for N, dfN in agg.groupby("N"):
    dfN = dfN.sort_values("alpha")
    x = dfN["alpha"].values
    y_mean = dfN["mean_branches"].values
    y_med  = dfN["median_branches"].values
    y_psat = dfN["prob_sat"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, y_mean, label="Mean branches", linewidth=2.2)
    ax1.plot(x, y_med,  '--', label="Median branches", linewidth=2.2)
    ax1.set_xlabel("L / N (alpha)")
    ax1.set_ylabel("Number of branches")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, y_psat, ':', label="Prob(SAT)", linewidth=2.2)
    ax2.set_ylabel("Prob(SAT)")
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title(f"Random 3-SAT (from o1 CNFs), CDCL, N = {N}")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"CDCL_from_o1_N_{N}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print("Saved figure:", fig_path)

print("Done.")
