import os
import re
import time
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Optional

# ===== Anthropic (Claude) SDK =====
import anthropic
from anthropic import Anthropic, RateLimitError

# ========= 基础配置（除调用方式外，其它保持不变） =========
# 建议用环境变量（更安全）： export ANTHROPIC_API_KEY=sk-ant-xxxx
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# 遍历的 Claude 模型列表（可按需增减）
model_list = [
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-3-opus-20240229",
]

# 输入输出目录（与你之前一致）
input_dir = "/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/cnf_results_openai_o1"
output_dir_base = "cnf_results_openai_o1_input_based_"  # 最终目录 = 该前缀 + 模型名

# 其它实验参数（保持不变）
N = 75
k = 3
alpha_values = np.arange(3.0, 6.0, 0.5)
instances_per_alpha = 20  # 仅用于计算 prob_sat 的分母（按你原逻辑保留）

# ========= 工具函数 =========
def safe_call_claude(client: Anthropic, prompt: str, model_selected: str) -> str:
    """
    使用 Anthropic Messages API 调用 Claude，带速率限制重试。
    """
    while True:
        try:
            resp = client.messages.create(
                model=model_selected,
                max_tokens=4096,
                temperature=0,
                system="You are a SAT solver. Follow the user's instructions exactly.",
                messages=[{"role": "user", "content": prompt}],
            )
            # 将所有 text block 拼接为字符串
            parts = []
            for block in resp.content:
                # 文本块通常为 type="text"
                if hasattr(block, "type") and block.type == "text":
                    parts.append(block.text)
                # 某些版本返回 dict 样式
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(p for p in parts if p is not None)
        except RateLimitError:
            print("Rate limit hit. Sleeping 20s...")
            time.sleep(20)
        except Exception as e:
            print("Unexpected error:", e)
            raise e

def cnf_to_prompt(clauses: List[List[int]]) -> str:
    lines = []
    for clause in clauses:
        parts = []
        for lit in clause:
            parts.append(f"x{lit}" if lit > 0 else f"¬x{-lit}")
        lines.append("(" + " ∨ ".join(parts) + ")")
    return "\n".join(lines)

def generate_k_sat(n_vars: int, n_clauses: int, k: int) -> List[List[int]]:
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = sample(range(1, n_vars + 1), k)
        clause = [var if randint(0, 1) else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses

def parse_response(text: str) -> Tuple[bool, int, int]:
    text = (text or "").lower()
    sat = "unsatisfiable" not in text
    branches = re.search(r"branches.*?(\d+)", text)
    conflicts = re.search(r"conflicts.*?(\d+)", text)
    b = int(branches.group(1)) if branches else 0
    c = int(conflicts.group(1)) if conflicts else 0
    return sat, b, c

def read_dimacs(filepath: str) -> List[List[int]]:
    clauses = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith(('c','p')):
                continue
            parts = line.strip().split()
            if 'time' in line.lower() and 'seconds' in line.lower():
                break
            if parts and parts[-1] == '0' and len(parts) == 4:
                clauses.append(list(map(int, parts[:-1])))
    return clauses

def parse_fname(fname: str) -> Tuple[int, float, int]:
    """
    解析：cnf_k{k}_N{N}_L{L}_alpha{alpha}_inst{inst}.txt
    返回 (L, alpha, inst_idx) 其中 inst_idx 为 0-based
    """
    base = os.path.basename(fname)
    m = re.match(
        r'^cnf_k(?P<k>\d+)_N(?P<N>\d+)_L(?P<L>\d+)_alpha(?P<alpha>\d+(?:\.\d+)?)_inst(?P<inst>\d+)\.txt$',
        base
    )
    if not m:
        raise ValueError(f"文件名不符合预期格式: {base}")
    L = int(m.group('L'))
    alpha = float(m.group('alpha'))
    inst = int(m.group('inst'))
    inst_idx = inst - 1
    return L, alpha, inst_idx

def build_prompt(cnf_text: str) -> str:
    # 与你原有提示词保持一致
    prompt = f"""You are a SAT logic solver.  
Please use a step-by-step method to solve the following 3-CNF formula.  

Finally, output only the following three items, with no extra explanation::
* Whether the formula is SATISFIABLE or UNSATISFIABLE  
* Number of branches (i.e., decision points)
* Number of conflicts (i.e., backtracking steps)
If the formula is SATISFIABLE, please give me the value for each literals.

The formula is:
{cnf_text}
"""
    return prompt

def should_skip_output(output_path: str) -> bool:
    """已有且无 error -> 跳过（保持你的原逻辑）"""
    if not os.path.exists(output_path):
        return False
    with open(output_path, "r", encoding="utf-8", errors='ignore') as rf:
        for line in rf:
            if "error" in line.lower():
                return False
    return True

def process_one_file(client: Anthropic, model: str, in_dir: str, out_dir: str, fname: str,
                     k_val: int, N_val: int) -> Optional[Tuple[bool, int, float]]:
    """
    处理单个文件：读取 -> prompt -> 调用 -> 解析 -> 写出结果文件
    返回 (sat, branches, elapsed_time)；如果跳过或解析失败返回 None
    """
    src = os.path.join(in_dir, fname)
    if not os.path.exists(src):
        return None

    out_path = os.path.join(out_dir, fname)
    if should_skip_output(out_path):
        return None

    cnf = read_dimacs(src)
    if not cnf:
        print(f"[WARN] no clauses parsed: {src}")
        return None

    cnf_text = cnf_to_prompt(cnf)
    prompt = build_prompt(cnf_text)

    try:
        start_time = time.time()
        response_text = safe_call_claude(client, prompt, model)
        elapsed_time = time.time() - start_time
        sat, branches, conflicts = parse_response(response_text)
    except Exception as e:
        response_text = f"[Error] {str(e)}"
        elapsed_time = 0
        sat, branches, conflicts = False, 0, 0  # 与原写入兼容

    # 输出文件名：沿用你的模板（N/k 来自全局；L/alpha/inst_idx 来自原文件名）
    L, alpha, inst_idx = parse_fname(fname)
    out_name = f"cnf_k{k_val}_N{N_val}_L{L}_alpha{round(alpha,2)}_inst{inst_idx+1}.txt"
    out_full = os.path.join(out_dir, out_name)
    with open(out_full, "w", encoding="utf-8") as f:
        f.write("c Random 3-SAT\n")
        f.write(f"c alpha={round(alpha,2)}, N={N_val}, L={L}, instance={inst_idx+1}\n")
        f.write(f"p cnf {N_val} {L}\n")
        for clause in cnf:
            f.write(" ".join(str(x) for x in clause) + " 0\n")
        f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
        f.write(response_text.strip())

    return (sat, branches, elapsed_time)

def run_for_model(client: Anthropic, model: str, in_dir: str, out_dir_base: str,
                  k_val: int, N_val: int, instances_per_alpha: int):
    """
    遍历 input_dir 中的所有 .txt，逐个生成输出，并做同样的“累加统计”。
    统计容器与原脚本一致：mean_branches/median_branches/prob_sat/avg_times 按处理进度不断追加。
    """
    out_dir = out_dir_base + model
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".txt")]
    files.sort()  # 稳定顺序

    instance_times: List[float] = []
    branches_list: List[int] = []
    sat_count = 0

    mean_branches: List[float] = []
    median_branches: List[float] = []
    prob_sat: List[float] = []
    avg_times: List[float] = []

    for fname in tqdm(files, desc=f"[scan inputs] ({model})"):
        res = process_one_file(client, model, in_dir, out_dir, fname, k_val, N_val)
        if res is None:
            continue
        sat, branches, elapsed = res

        instance_times.append(elapsed)
        if sat:
            sat_count += 1
        branches_list.append(branches)

        # 与你原脚本一致：每处理一个文件就 append 当前统计
        mean_branches.append(mean(branches_list))
        median_branches.append(median(branches_list))
        avg_times.append(mean(instance_times))
        prob_sat.append(sat_count / instances_per_alpha)

    print(f"\n[MODEL={model}] files_processed={len(branches_list)} "
          f"mean_br={mean(branches_list) if branches_list else 0:.2f} "
          f"sat_cnt={sat_count}\n")

    return mean_branches, median_branches, prob_sat, avg_times

# ========= 主流程：遍历 Claude 模型 =========
def main():
    if not ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY.startswith("sk-ant-"):
        raise RuntimeError("请设置有效的 ANTHROPIC_API_KEY（或直接在脚本中替换为你的 key）。")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    all_results = {}
    for model in model_list:
        print(f"\n==== Running for model: {model} ====")
        mb, medb, ps, avgt = run_for_model(
            client=client,
            model=model,
            in_dir=input_dir,
            out_dir_base=output_dir_base + '_',  # 最终目录：cnf_results_openai_o1_input_based__<model>
            k_val=k,
            N_val=N,
            instances_per_alpha=instances_per_alpha
        )
        all_results[model] = {
            "mean_branches": mb,
            "median_branches": medb,
            "prob_sat": ps,
            "avg_times": avgt,
        }

if __name__ == "__main__":
    main()
