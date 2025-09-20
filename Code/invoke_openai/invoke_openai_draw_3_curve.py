import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re

# 设置 OpenAI API KEY
# openai.api_key = os.environ["OPENAI_API_KEY"]  # 你的API KEY

openai.api_key = os.environ["OPENAI_API_KEY"]

openai.api_base = "https://api.openai.com/v1"
# openai.api_base = 'https://api.openai.com/v1/responses'
# ['gpt-4-0613', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o-audio-preview-2025-06-03', 'gpt-4.1-nano-2025-04-14', 'gpt-4.1-nano', 'gpt-image-1', 'gpt-4o-realtime-preview-2025-06-03', 'davinci-002', 'babbage-002', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct-0914', 'dall-e-3', 'dall-e-2', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'tts-1-hd', 'tts-1-1106', 'tts-1-hd-1106', 'text-embedding-3-small', 'text-embedding-3-large', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'chatgpt-4o-latest', 'o1-preview-2024-09-12', 'o1-preview', 'o1-mini-2024-09-12', 'o1-mini', 'gpt-4o-realtime-preview-2024-10-01', 'gpt-4o-audio-preview-2024-10-01', 'gpt-4o-audio-preview', 'gpt-4o-realtime-preview', 'omni-moderation-latest', 'omni-moderation-2024-09-26', 'gpt-4o-realtime-preview-2024-12-17', 'gpt-4o-audio-preview-2024-12-17', 'gpt-4o-mini-realtime-preview-2024-12-17', 'gpt-4o-mini-audio-preview-2024-12-17', 'o1-2024-12-17', 'o1', 'gpt-4o-mini-realtime-preview', 'gpt-4o-mini-audio-preview', 'o3-mini', 'o3-mini-2025-01-31', 'gpt-4o-2024-11-20', 'gpt-4.5-preview', 'gpt-4.5-preview-2025-02-27', 'gpt-4o-search-preview-2025-03-11', 'gpt-4o-search-preview', 'gpt-4o-mini-search-preview-2025-03-11', 'gpt-4o-mini-search-preview', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'o1-pro-2025-03-19', 'o1-pro', 'gpt-4o-mini-tts', 'gpt-4.1-2025-04-14', 'gpt-4.1', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-mini', 'gpt-3.5-turbo-16k', 'tts-1', 'whisper-1', 'text-embedding-ada-002']
model_selected = 'gpt-4.1' # 'gpt-4-turbo' # 'chatgpt-4o-latest'  #'gpt-4.1' # 'gpt-4o' #'gpt-3.5-turbo'  'gpt-3.5-turbo-0125'    'o1'
# o3,  o4-mini,

# o1-pro, o3, o3-mini, o3-pro, o3-deep-research, o4-mini,
# 输出目录

input_dir = '/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/cnf_results_openai_o1'
output_dir = 'cnf_results_openai_o1_input_based_' + model_selected
os.makedirs(output_dir, exist_ok=True)
# 参数设置
N = 75
k = 3
alpha_values = np.arange(3.0, 6.0, 0.5)
# alpha_values = np.arange(1.0, 4.5, 0.5)

instances_per_alpha = 20  # 测试阶段建议值小

api_key = os.environ["OPENAI_API_KEY"]

api_base = "https://api.openai.com/v1"
# Use OpenAI v1 interface
client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)

# 结果记录容器
mean_branches = []
median_branches = []
prob_sat = []
avg_times = []

# 安全调用ChatGPT（含限速处理）
def safe_call_chatgpt(prompt, model_selected):

    # while True:
    try:
        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=model_selected,
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate limit hit. Sleeping 20s...")
            time.sleep(20)
        else:
            print("Unexpected error:", e)
            raise e

# CNF 转为字符串
def cnf_to_prompt(clauses):
    lines = []
    for clause in clauses:
        parts = []
        for lit in clause:
            parts.append(f"x{lit}" if lit > 0 else f"¬x{-lit}")
        lines.append("(" + " ∨ ".join(parts) + ")")
    return "\n".join(lines)

# 随机生成 k-SAT CNF
def generate_k_sat(n_vars, n_clauses, k):
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = sample(range(1, n_vars + 1), k)
        clause = [var if randint(0, 1) else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses

# 解析 ChatGPT 响应
def parse_response(text):
    text = text.lower()
    sat = "unsatisfiable" not in text
    branches = re.search(r"branches.*?(\d+)", text)
    conflicts = re.search(r"conflicts.*?(\d+)", text)
    b = int(branches.group(1)) if branches else 0
    c = int(conflicts.group(1)) if conflicts else 0
    return sat, b, c


def read_dimacs(filepath):
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


def parse_fname(fname):
    """
    解析形如：
    cnf_k{k}_N{N}_L{L}_alpha{alpha}_inst{inst}.txt
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
    alpha = float(m.group('alpha'))        # 支持 4.2 / 4.20 等
    inst = int(m.group('inst'))            # 文件名里的 inst = inst_idx + 1
    inst_idx = inst - 1                    # 还原成 0-based

    return L, alpha, inst_idx

files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]

instance_times = []
branches_list = []
sat_count = 0
written_total = 0
for fname in tqdm(files, desc="[scan inputs]"):
    src = os.path.join(input_dir, fname)
    if not os.path.exists(src): continue

    out_path = os.path.join(output_dir, fname)

    # 已有且无 error → 跳过
    skip = False
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", errors='ignore') as rf:
            for line in rf:
                if "error" in line.lower():
                    skip = False
                    break
            else:
                skip = True
    if skip: continue

    cnf = read_dimacs(src)
    if not cnf:
        print(f"[WARN] no clauses parsed: {src}")
        continue

    cnf_text = cnf_to_prompt(cnf)

    #         prompt = f"""You are a SAT logic solver.
    # Please use a step-by-step method to solve the following 3-CNF formula.
    #
    # At each step, record:
    # * Which variable is assigned
    # * Any propagated implications
    # * Whether the formula is satisfied or a conflict occurs
    #
    # Finally, output:
    # * Whether the formula is SATISFIABLE or UNSATISFIABLE
    # * Number of branches (i.e., decision points)
    # * Number of conflicts (i.e., backtracking steps)
    #
    # The formula is:
    # {cnf_text}
    # """

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

    # 调用模型 + 记录时间
    try:
        start_time = time.time()
        response = safe_call_chatgpt(prompt, model_selected)
        elapsed_time = time.time() - start_time
        sat, branches, conflicts = parse_response(response)
    except Exception as e:
        response = f"[Error] {str(e)}"
        elapsed_time = 0
        sat, branches, conflicts = "Call_API_Error", 0, 0

    instance_times.append(elapsed_time)
    if sat:
        sat_count += 1
    branches_list.append(branches)

    L, alpha, inst_idx = parse_fname(fname)
    # 写入文件
    fname = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha,2)}_inst{inst_idx+1}.txt"
    with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
        f.write("c Random 3-SAT\n")
        f.write(f"c alpha={round(alpha,2)}, N={N}, L={L}, instance={inst_idx+1}\n")
        f.write(f"p cnf {N} {L}\n")
        for clause in cnf:
            f.write(" ".join(str(x) for x in clause) + " 0\n")
        f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
        f.write(response.strip())

    # 每个 alpha 统计汇总
    mean_branches.append(mean(branches_list))
    median_branches.append(median(branches_list))
    avg_times.append(mean(instance_times))
    prob_sat.append(sat_count / instances_per_alpha)

