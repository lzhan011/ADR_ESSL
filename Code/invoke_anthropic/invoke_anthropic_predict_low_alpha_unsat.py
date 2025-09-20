import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re
import anthropic

api_key = os.environ["ANTHROPIC_API_KEY"]
client = anthropic.Anthropic(api_key=api_key)
model_selected = "claude-3-5-haiku-20241022" # "claude-3-7-sonnet-20250219" # "claude-sonnet-4-20250514"   # "claude-3-opus-20240229"   claude-opus-4-20250514
N_list = [5, 60,  8, 10, 25, 50]  # 5



k = 3
alpha_values = [3.5, 3.6, 3.7, 3.8, 3.9,4.0, 4.1, 4.2]  # 小 alpha，更难得到 UNSAT

instances_per_alpha = 10000  # 测试阶段建议值小

# 结果记录容器
mean_branches = []
median_branches = []
prob_sat = []
avg_times = []




def safe_call_claude(prompt):
    try:
        response = client.messages.create(
            model= model_selected,  # or another Claude 3 model
            max_tokens=4096,
            temperature=0,
            system="You are a SAT solver. Follow the user's instructions exactly.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
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
            if parts and parts[-1] == '0':
                clause = list(map(int, parts[:-1]))
                clauses.append(clause)
    return clauses


# 主循环


for N in N_list:
    input_dir = 'unsat_cnf_low_alpha_N_' + str(N)
    output_dir = input_dir + "_anthropic_prediction_" + str(model_selected)
    os.makedirs(output_dir, exist_ok=True)

    write_file_number = 0
    for alpha in tqdm(alpha_values, desc="Processing alpha values"):
        L = int(alpha * N)
        branches_list = []
        instance_times = []
        sat_count = 0


        for inst_idx in range(instances_per_alpha):
            # cnf = generate_k_sat(N, L, k)
            # 保存该 UNSAT 的 CNF 文件
            filename = f"UNSAT_k{k}_N{N}_L{L}_alpha{round(alpha, 2)}_inst{inst_idx + 1}.cnf"
            filepath = os.path.join(input_dir, filename)

            if not os.path.exists(filepath):
                continue
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
                response = safe_call_claude(prompt)
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
                write_file_number += 1
                print("write_file_number:", write_file_number)

            if write_file_number >= 70:
                break

        # 每个 alpha 统计汇总
        if len(branches_list)==0:
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
