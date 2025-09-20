import os, re, time, json
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Anthropic ===
import anthropic
api_key = os.environ["ANTHROPIC_API_KEY"]
client = anthropic.Anthropic(api_key=api_key)

#model_selected = "claude-3-5-haiku-20241022" # "claude-3-7-sonnet-20250219" # "claude-sonnet-4-20250514"   # "claude-3-opus-20240229"   claude-opus-4-20250514

# 选一个 Claude 模型
model_selected = "claude-3-5-haiku-20241022"
# 或者：
anthropic_models = ["claude-opus-4-20250514","claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219",  "claude-3-opus-20240229" , ]
anthropic_models = ["claude-opus-4-20250514"]

def safe_call_claude(prompt):
    # while True:
    try:
        response = client.messages.create(
            model=model_selected,
            max_tokens=4096,
            temperature=0,
            system="You are a SAT solver. Follow the user's instructions exactly.",
            messages=[{"role": "user", "content": prompt}]
        )
        return "".join(
            (block.text for block in response.content if getattr(block, "type", None) == "text")
        ) or (response.content[0].text if response.content else "")
    except anthropic.RateLimitError:
        print("Rate limit hit. Sleeping 20s...")
        time.sleep(20)
    except Exception as e:
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

O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL'

# 主循环里的调用改为：
for model_selected in anthropic_models:   # 如果你用多模型列表
    for N in [5, 8, 10, 25, 50]:
        N = str(N)
        dir_name = f"cnf_results_CDCL_N_{N}"
        O1_input_dir = os.path.join(O1_input_dir_root, dir_name)
        output_dir = os.path.join(O1_input_dir_root, 'prediction_result',
                                  dir_name + '_openai_prediction_' + str(model_selected))  # 这里你也可以把 openai 改为 anthropic
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        write_file_number = 0
        for file in os.listdir(O1_input_dir):
            filepath = os.path.join(O1_input_dir, file)
            output_path = os.path.join(output_dir, file)
            if not os.path.exists(filepath):
                continue

            error_signal = False
            if os.path.exists(output_path):
                with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        if "error" in line.lower():
                            error_signal = True
                            break
            if os.path.exists(output_path) and not error_signal:
                continue

            cnf = read_dimacs(filepath)
            cnf_text = cnf_to_prompt(cnf)
            prompt = make_sat_json_prompt_2cnf(cnf_text, N)

            try:
                start_time = time.time()
                response = safe_call_claude(prompt)   # <=== 改这里
                elapsed_time = time.time() - start_time
                sat, branches, conflicts = parse_response(response)
            except Exception as e:
                response = f"[Error] {str(e)}"
                elapsed_time = 0
                sat, branches, conflicts = "Call_API_Error", 0, 0

            N_val, L, alpha, inst_idx = find_k_n_alpha(file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("c Random 2-SAT\n")
                f.write(f"c alpha={round(alpha, 2)}, N={N_val}, L={L}, instance={inst_idx + 1}\n")
                f.write(f"p cnf {N_val} {L}\n")
                for clause in cnf:
                    f.write(" ".join(str(x) for x in clause) + " 0\n")
                f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
                f.write(response.strip())
                write_file_number += 1
                print("write_file_number:", write_file_number)
