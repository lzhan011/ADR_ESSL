import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re
from openai import OpenAI, RateLimitError

# ----------------- 基本配置 -----------------

model_list = ['gpt-5']  # 你想跑的模型列表
api_key = os.environ.get("OPENAI_API_KEY")
api_base = "https://api.openai.com/v1"

client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)

# ----------------- 调用 ChatGPT 的封装 -----------------

def safe_call_chatgpt(prompt, model_selected):
    """带限速重试的 ChatGPT 调用"""
    while True:
        try:
            messages = [{"role": "user", "content": prompt}]
            resp = client.chat.completions.create(
                model=model_selected,
                messages=messages
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg:
                print("Rate limit hit. Sleeping 60s...")
                time.sleep(60)
                continue
            else:
                print("Unexpected error:", e)
                raise e

# ----------------- CNF 相关工具函数 -----------------

def cnf_to_prompt(clauses):
    """把 CNF 子句转换成可读的公式字符串"""
    lines = []
    for clause in clauses:
        if not clause:
            # 空子句
            lines.append("()")
            continue
        parts = []
        for lit in clause:
            parts.append(f"x{lit}" if lit > 0 else f"¬x{-lit}")
        lines.append("(" + " ∨ ".join(parts) + ")")
    return "\n".join(lines)


def parse_response(text):
    """解析模型输出：判断 SAT/UNSAT，并提取 branches/conflicts（如果有的话）"""
    text_low = text.lower()
    # 简单规则：包含 unsatisfiable 就当 UNSAT
    sat = "unsatisfiable" not in text_low

    branches = re.search(r"branches.*?(\d+)", text_low)
    conflicts = re.search(r"conflicts.*?(\d+)", text_low)
    b = int(branches.group(1)) if branches else 0
    c = int(conflicts.group(1)) if conflicts else 0
    return sat, b, c


def read_dimacs(filepath):
    """
    通用 DIMACS 解析：
    返回 (num_vars, num_clauses_header, clauses)
    - clauses: List[List[int]]，每个子句不含结尾 0
    """
    clauses = []
    num_vars = None
    num_clauses_header = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                # p cnf <num_vars> <num_clauses>
                if len(parts) >= 4 and parts[1] == 'cnf':
                    num_vars = int(parts[2])
                    num_clauses_header = int(parts[3])
                continue

            # 子句行：一直读到 0 为止
            parts = line.split()
            lits = []
            for p in parts:
                lit = int(p)
                if lit == 0:
                    break
                lits.append(lit)
            if lits:
                clauses.append(lits)

    if num_vars is None:
        # 如果缺少 p cnf 行，就粗糙地根据最大变量号估计一下
        max_var = 0
        for cl in clauses:
            for lit in cl:
                max_var = max(max_var, abs(lit))
        num_vars = max_var
    if num_clauses_header is None:
        num_clauses_header = len(clauses)

    return num_vars, num_clauses_header, clauses




def generate_prompt(cnf_text):
    prompt = f"""
    You are an expert in Boolean satisfiability (SAT).  
    Given the following CNF formula in conjunctive normal form, determine whether it is SATISFIABLE.

    OUTPUT FORMAT (VERY IMPORTANT):
    Return ONLY **one single-line JSON object** with keys:
      - "answer": either "SAT" or "UNSAT"
      - "assignment": a list of integers representing a complete satisfying assignment **if and only if** answer="SAT";
                      otherwise []
      - "explain": a short plain-text explanation (1–2 sentences)

    Rules:
      * Output "SAT" ONLY IF you can explicitly provide a complete integer assignment that satisfies **all clauses**.
      * If you are unsure OR cannot check all clauses OR cannot list a complete assignment, return "UNSAT" and assignment=[].
      * The JSON must be valid and appear **on a single line**.
      * No additional text before or after the JSON.
      * A positive literal xN means variable N=True; negative literal -xN means variable N=False.

    Example valid outputs:
      {{"answer": "SAT", "assignment": [1, -2, 3, -4], "explain": "All clauses satisfied by the assignment."}}
      {{"answer": "UNSAT", "assignment": [], "explain": "Unable to find a satisfying assignment."}}

    Now analyze the CNF formula below:

    {cnf_text}
    """

    return prompt
# ----------------- 主逻辑：pair-by-pair 调用 -----------------

if __name__ == "__main__":
    # flat50-115 目录，包含 SAT 和 *_UNSAT.cnf
    input_dir_root = 'Flat_Graph_Colouring/flat50-115'
    output_dir = 'Flat_Graph_Colouring'
    for model_selected in model_list:
        print(f"\n=== Running model: {model_selected} ===")

        input_dir = input_dir_root
        # 每个模型一个输出目录
        output_dir = os.path.join(output_dir, f"flat50-115_openai_prediction_{model_selected}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        write_file_number = 0

        # 找到所有“base”文件：.cnf 且不包含 _UNSAT
        all_files = sorted(os.listdir(input_dir))
        base_files = [
            f for f in all_files
            if f.endswith(".cnf") and "_UNSAT" not in f
        ]
        ii = 0
        for base_file in base_files:
            ii = ii + 1
            if ii >= 30:
                break
            sat_file = base_file
            unsat_file = os.path.splitext(base_file)[0] + "_UNSAT.cnf"

            sat_path = os.path.join(input_dir, sat_file)
            unsat_path = os.path.join(input_dir, unsat_file)

            if not os.path.exists(unsat_path):
                print(f"[WARN] UNSAT pair not found for {sat_file}, expected {unsat_file}, skip.")
                continue

            # 一对一对地处理：先 SAT 再 UNSAT
            for label, filepath in [("SAT", sat_path), ("UNSAT", unsat_path)]:
                file_name = os.path.basename(filepath)
                # 输出文件名：比如 flat30-1_SAT_gpt-5.txt
                out_name = f"{os.path.splitext(file_name)[0]}_{label}_{model_selected}.txt"
                output_path = os.path.join(output_dir, out_name)

                # 如果输出已经存在且不包含 error，就跳过
                error_signal = False
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "error" in line.lower():
                                error_signal = True
                                break
                if os.path.exists(output_path) and not error_signal:
                    # 已有正常结果，跳过
                    continue

                # 读 CNF
                num_vars, num_clauses_header, cnf = read_dimacs(filepath)
                cnf_text = cnf_to_prompt(cnf)

                prompt = generate_prompt(cnf_text)


                # 调用模型 + 记录时间
                try:
                    start_time = time.time()
                    response = safe_call_chatgpt(prompt, model_selected)
                    elapsed_time = time.time() - start_time
                    sat_pred, branches, conflicts = parse_response(response)
                except Exception as e:
                    response = f"[Error] {str(e)}"
                    elapsed_time = 0
                    sat_pred, branches, conflicts = "Call_API_Error", 0, 0

                # 写入结果文件
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"c Flat graph colouring CNF, true_label={label}\n")
                    f.write(f"c model={model_selected}\n")
                    f.write(f"p cnf {num_vars} {len(cnf)}\n")
                    for clause in cnf:
                        f.write(" ".join(str(x) for x in clause) + " 0\n")
                    f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n")
                    f.write(f"c predicted_sat={sat_pred}, branches={branches}, conflicts={conflicts}\n\n")
                    f.write(response.strip())

                write_file_number += 1
                print(f"[{model_selected}] {label} file done:", output_path, "  (#", write_file_number, ")")
