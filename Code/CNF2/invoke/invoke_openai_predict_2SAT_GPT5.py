import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re
from openai import OpenAI
from openai import OpenAI, RateLimitError

# 设置 API KEY

model_list = [ 'gpt-3.5-turbo', 'o3-mini','gpt-4o' , 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', ]  # 可以替换成你想跑的模型
model_list = ['gpt-5']
model_list = ['chatgpt-4o-latest']
# model_selected = "o1"

api_key = os.environ["OPENAI_API_KEY"]

api_base = "https://api.openai.com/v1"
# Use OpenAI v1 interface
client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)

k = 3
alpha_values = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]  # 小 alpha，更难得到 UNSAT

instances_per_alpha = 10000  # 测试阶段建议值小

# 结果记录容器
mean_branches = []
median_branches = []
prob_sat = []
avg_times = []


# 安全调用DeepSeek（含限速处理）
def safe_call_deepseek(prompt):
    while True:
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
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            parts = line.strip().split()
            # if parts and parts[-1] == '0':
            if parts and parts[-1] == '0' and len(parts) == 3 and re.fullmatch(r"[\d\-]", line[0]):
                clause = list(map(int, parts[:-1]))
                clauses.append(clause)
    return clauses


def find_k_n_alpha(file_name):
    import re

    # fname = "cnf_k3_N100_L50_alpha4.27_inst23.txt"

    # 使用正则表达式查找所有匹配的字段
    matches = re.findall(r"N(\d+)|L(\d+)|alpha([\d.]+)|inst(\d+)", file_name)

    # 初始化变量
    N = L = alpha = inst_idx = None

    # 遍历匹配结果并赋值
    for n, l, a, i in matches:
        if n:
            N = int(n)
        if l:
            L = int(l)
        if a:
            alpha = float(a)
        if i:
            inst_idx = int(i)

    print(f"N = {N}, L = {L}, alpha = {alpha}, inst_idx = {inst_idx}")

    # return {"N":N, "L":L, "alpha":alpha, "inst_idx":inst_idx}
    return N, L, alpha, inst_idx



def make_sat_json_prompt_2cnf(cnf_text: str, n_vars) -> str:
    """
    将 2-CNF 公式包装为严格 JSON 输出的提示词。

    参数:
        cnf_text: 已格式化的 2-CNF 文本（如 DIMACS 或自定义格式），会直接嵌入到提示中。
        n_vars:   可选。变量总数 N；若为 None，则令 N 为公式中出现的最大变量索引。
        instance_name: 可选。实例名称，便于区分运行样本。
    返回:
        prompt 字符串
    """
    n_hint = f"(N = {n_vars}) " if n_vars is not None else "(N equals the largest index appearing in the formula) "
    header = (
        "You are an expert SAT solver. "
        "Determine the satisfiability of the following 2-CNF formula.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "Return ONLY a single-line JSON object with keys:\n"
        '  - "answer": either "SATISFIABLE" or "UNSATISFIABLE"\n'
        '  - "branches": a non-negative integer for decision points\n'
        '  - "conflicts": a non-negative integer for backtracking conflicts\n'
        '  - "assignment": a list of signed integers representing a COMPLETE assignment iff "answer" is "SATISFIABLE"; otherwise []\n'
        "Rules:\n"
        "  * Do any step-by-step reasoning internally; DO NOT include reasoning or explanations in the output.\n"
        f"  * If SATISFIABLE, output a total assignment covering variables 1..N {n_hint}"
        "    where +i means xi = True and -i means xi = False. Include each i exactly once; exclude 0.\n"
        "  * If UNSATISFIABLE or unsure OR you cannot verify a complete assignment, set assignment = [].\n"
        "  * No text outside the JSON. One line only.\n"
        "Example valid outputs:\n"
        '  {{\"answer\":\"SATISFIABLE\",\"branches\":2,\"conflicts\":0,\"assignment\":[1,-2,3]}}\n'
        '  {{\"answer\":\"UNSATISFIABLE\",\"branches\":5,\"conflicts\":3,\"assignment\":[]}}\n'
    )

    body = (
        "Formula (2-CNF):\n"
        f"{cnf_text}\n\n"
        "Task: Decide satisfiability and return only the JSON described above."
    )
    return header + "\n" + body



# 主循环
# O1_input_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_openai\draw_o1_phase_transition_figures\draw_o1_cnf_alpha_3_6_N_75'
#
# O1_input_dir = r'/work/lzhan011/Satisfiability_Solvers/Code/invoke_openai/draw_o1_phase_transition_figures/draw_o1_cnf_alpha_3_6_N_75'

O1_input_dir_root = r'fixed_set_mul_N'
O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL'

for model_selected in model_list:
    # for N in [5,8,10,25,50,60]:
    # for N in [5, 8, 10, 25, 50, 60]:
    for N in [25]:

        N = str(N)
        dir_name = f"cnf_results_CDCL_N_{N}"
        O1_input_dir = os.path.join(O1_input_dir_root, dir_name)
        output_dir = os.path.join(O1_input_dir_root, 'prediction_result',
                                  dir_name + '_openai_prediction_' + str(model_selected))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        write_file_number = 0

        for file in os.listdir(O1_input_dir):
            filepath = os.path.join(O1_input_dir, file)
            output_path = os.path.join(output_dir, file)
            if not os.path.exists(filepath):
                continue

            # if os.path.exists(output_path):
            #     continue

            error_signal = False
            if os.path.exists(output_path):
                with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        if "error" in line.lower():
                            error_signal = True
                            break
            if  os.path.exists(output_path) :
                if error_signal:
                    pass
                else:
                    continue
            else:
                pass

            cnf = read_dimacs(filepath)
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

            # prompt = f"""You are a SAT logic solver.
            # Please use a step-by-step method to solve the following 3-CNF formula.
            #
            # Finally, output only the following three items, with no extra explanation::
            # * Whether the formula is SATISFIABLE or UNSATISFIABLE
            # * Number of branches (i.e., decision points)
            # * Number of conflicts (i.e., backtracking steps)
            # If the formula is SATISFIABLE, please give me the value for each literals.
            #
            # The formula is:
            # {cnf_text}
            # """
            prompt = make_sat_json_prompt_2cnf(cnf_text, N)

            # 调用模型 + 记录时间
            try:
                start_time = time.time()
                response = safe_call_deepseek(prompt)
                elapsed_time = time.time() - start_time
                sat, branches, conflicts = parse_response(response)
            except Exception as e:
                response = f"[Error] {str(e)}"
                elapsed_time = 0
                sat, branches, conflicts = "Call_API_Error", 0, 0

            # 写入文件

            N, L, alpha, inst_idx = find_k_n_alpha(file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("c Random 2-SAT\n")
                f.write(f"c alpha={round(alpha, 2)}, N={N}, L={L}, instance={inst_idx + 1}\n")
                f.write(f"p cnf {N} {L}\n")
                for clause in cnf:
                    f.write(" ".join(str(x) for x in clause) + " 0\n")
                f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
                f.write(response.strip())
                write_file_number += 1
                print("write_file_number:", write_file_number)



