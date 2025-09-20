import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re

# pip install openai==0.28


# 设置 OpenAI API KEY
# openai.api_key = os.environ["OPENAI_API_KEY"]  # 你的API KEY

api_key = os.environ["OPENAI_API_KEY"]

api_base = "https://api.openai.com/v1"
# Use OpenAI v1 interface
client = openai.OpenAI(
    api_key=api_key,
    base_url=api_base
)
# ['gpt-4-0613', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o-audio-preview-2025-06-03', 'gpt-4.1-nano-2025-04-14', 'gpt-4.1-nano', 'gpt-image-1', 'gpt-4o-realtime-preview-2025-06-03', 'davinci-002', 'babbage-002', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct-0914', 'dall-e-3', 'dall-e-2', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'tts-1-hd', 'tts-1-1106', 'tts-1-hd-1106', 'text-embedding-3-small', 'text-embedding-3-large', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'chatgpt-4o-latest', 'o1-preview-2024-09-12', 'o1-preview', 'o1-mini-2024-09-12', 'o1-mini', 'gpt-4o-realtime-preview-2024-10-01', 'gpt-4o-audio-preview-2024-10-01', 'gpt-4o-audio-preview', 'gpt-4o-realtime-preview', 'omni-moderation-latest', 'omni-moderation-2024-09-26', 'gpt-4o-realtime-preview-2024-12-17', 'gpt-4o-audio-preview-2024-12-17', 'gpt-4o-mini-realtime-preview-2024-12-17', 'gpt-4o-mini-audio-preview-2024-12-17', 'o1-2024-12-17', 'o1', 'gpt-4o-mini-realtime-preview', 'gpt-4o-mini-audio-preview', 'o3-mini', 'o3-mini-2025-01-31', 'gpt-4o-2024-11-20', 'gpt-4.5-preview', 'gpt-4.5-preview-2025-02-27', 'gpt-4o-search-preview-2025-03-11', 'gpt-4o-search-preview', 'gpt-4o-mini-search-preview-2025-03-11', 'gpt-4o-mini-search-preview', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'o1-pro-2025-03-19', 'o1-pro', 'gpt-4o-mini-tts', 'gpt-4.1-2025-04-14', 'gpt-4.1', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-mini', 'gpt-3.5-turbo-16k', 'tts-1', 'whisper-1', 'text-embedding-ada-002']
model_selected = 'o1' # 'gpt-4-turbo' # 'chatgpt-4o-latest'  #'gpt-4.1' # 'gpt-4o' #'gpt-3.5-turbo'  'gpt-3.5-turbo-0125'    'o1'
# o3,  o4-mini,
model_list = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', 'gpt-4o', ]  # 可以替换成你想跑的模型
# o1-pro, o3, o3-mini, o3-pro, o3-deep-research, o4-mini,
# 输出目录
# output_dir = "cnf_results_openai_"+model_selected + "_small_alpha" + "_set_literals_value"
# N = 5

N_list = [5, 8, 10, 25, 50, 60,70, 80, 90, 100, 110, 120, 130, 140]
# N_list = [70]
k = 3
alpha_values = [3.5, 3.6, 3.7, 3.8, 3.9,4.0, 4.1, 4.2]  # 小 alpha，更难得到 UNSAT

instances_per_alpha = 10000  # 测试阶段建议值小

# 结果记录容器
mean_branches = []
median_branches = []
prob_sat = []
avg_times = []

# 安全调用ChatGPT（含限速处理）
def safe_call_chatgpt(prompt, model_selected):
    while True:
        try:
            if model_selected in ['o1', 'o3-mini']:
                #     response = openai.ChatCompletion.create(
                #         model=model_selected,
                #         messages=[{"role": "user", "content": prompt}],
                #     )
                #     return response['choices'][0]['message']['content']
                response = client.chat.completions.create(
                    model=model_selected,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    # max_tokens=4096,
                    # stop=["END."]
                )
                return response.choices[0].message.content

                # messages = [{"role": "user", "content": prompt}]
                # resp = client.chat.completions.create(
                #     model=model_selected,
                #     messages=messages
                # )
                # return resp.choices[0].message.content
            else:

                # response = openai.ChatCompletion.create(
                #     model=model_selected,
                #     messages=[{"role": "user", "content": prompt}],
                #     temperature=0,
                #     max_tokens=4096,
                #     stop=["END."]
                # )
                # return response['choices'][0]['message']['content']

                response = client.chat.completions.create(
                    model=model_selected,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                    stop=["END."]
                )
                return response.choices[0].message.content

        except openai.error.RateLimitError:
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
    print("filepath:", filepath)
    clauses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            parts = line.strip().split()
            if parts and parts[-1] == '0' and len(parts) == 4 and re.fullmatch(r"[\d\-]", line[0]):
                clause = list(map(int, parts[:-1]))
                clauses.append(clause)
    return clauses


# 主循环

# 主循环（对每个模型跑一遍）
for model_selected in model_list:
    mean_branches = []
    median_branches = []
    prob_sat = []
    avg_times = []

    for N in N_list:
        input_dir = f'unsat_cnf_low_alpha_N_{N}'
        input_dir_o1 = f'unsat_cnf_low_alpha_N_{N}_openai_prediction_o1'
        output_dir = f"{input_dir}_openai_prediction_{model_selected}"
        print("output_dir:", output_dir)

        os.makedirs(output_dir, exist_ok=True)

        write_file_number = 0
        for alpha in tqdm(alpha_values, desc=f"Model {model_selected} | alpha"):
            L = int(alpha * N)
            branches_list = []
            instance_times = []
            sat_count = 0

            for inst_idx in range(instances_per_alpha):
                filename = f"UNSAT_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}.cnf"

                f_o1_output_name = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}.txt"
                f_input_filepath = os.path.join(input_dir_o1, f_o1_output_name)
                if not os.path.exists(f_input_filepath):
                    continue

                f_o1_output_name = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}.txt"
                output_file = os.path.join(output_dir, f_o1_output_name)

                if os.path.exists(output_file):
                    error_flag_exist = False
                    with open(f_input_filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'error' in line.lower():
                                error_flag_exist = True
                                break
                    if error_flag_exist:
                        pass
                    else:
                        continue

                cnf = read_dimacs(f_input_filepath)
                cnf_text = cnf_to_prompt(cnf)

                prompt = f"""You are a SAT logic solver.  
Please use a step-by-step method to solve the following 3-CNF formula.  

Finally, output only the following three items, with no extra explanation:
* Whether the formula is SATISFIABLE or UNSATISFIABLE  
* Number of branches (i.e., decision points)
* Number of conflicts (i.e., backtracking steps)
If the formula is SATISFIABLE, please give me the value for each literal.

The formula is:
{cnf_text}
"""

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

                # fname = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha,2)}_inst{inst_idx+1}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("c Random 3-SAT\n")
                    f.write(f"c alpha={round(alpha,2)}, N={N}, L={L}, instance={inst_idx+1}\n")
                    f.write(f"p cnf {N} {L}\n")
                    for clause in cnf:
                        f.write(" ".join(str(x) for x in clause) + " 0\n")
                    f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
                    f.write(response.strip())
                    write_file_number += 1
                    print("write_file_number:", write_file_number)

                if write_file_number >= 70:
                    break

            if len(branches_list) == 0:
                continue
            mean_branches.append(mean(branches_list))
            median_branches.append(median(branches_list))
            avg_times.append(mean(instance_times))
            prob_sat.append(sat_count / instances_per_alpha)

            if write_file_number >= 70:
                break

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 主坐标轴：分支数量
ax1.plot(alpha_values, mean_branches, label="Mean branches", color="black")
ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="black")
line1, = ax1.plot(alpha_values, mean_branches, label="Mean branches", color="black")
line2, = ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="black")
ax1.set_xlabel("L / N")
ax1.set_ylabel("Number of branches")
ax1.legend(loc="upper left")

# 第二坐标轴：可满足性概率
ax2 = ax1.twinx()
ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="blue")
ax2.set_ylabel("Prob(sat)")
line3, = ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="blue")
ax2.legend(loc="lower left")

# 第三坐标轴：平均耗时
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(alpha_values, avg_times, '-.', label="Avg solve time (s)", color="gray")
ax3.set_ylabel("Avg solve time (seconds)")
line4, = ax3.plot(alpha_values, avg_times, '-.', label="Avg solve time (s)", color="gray")
ax3.legend(loc="upper right")

lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)  # 合并图例放在下方


plt.title("OpenAI GPT-3.5 on Random 3-SAT, N=75")
plt.grid(True)
plt.tight_layout()
plt.show()
