import openai
import os
import numpy as np
from random import randint, sample
from statistics import mean, median
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置 API KEY
openai.api_key = os.environ["DEEPSEEK_API_KEY"]
openai.api_base = "https://api.deepseek.com/v1"
model_selected =  "deepseek-chat"  # 或 deepseek-coder

# 设置参数
N = 5
k = 3
alpha_values = np.arange(3.0, 6.0, 0.5)
instances_per_alpha = 20  # DeepSeek API耗费较大，建议降低
# 输出目录
output_dir = "cnf_results_deepseek_N_" + str(N) + "_model_selected_" + str(model_selected)
os.makedirs(output_dir, exist_ok=True)



# 结果容器
mean_branches = []
median_branches = []
prob_sat = []

# CNF 转成 prompt 字符串
def cnf_to_prompt(clauses):
    lines = []
    for clause in clauses:
        parts = []
        for lit in clause:
            if lit > 0:
                parts.append(f"x{lit}")
            else:
                parts.append(f"¬x{-lit}")
        lines.append("(" + " ∨ ".join(parts) + ")")
    return "\n".join(lines)

# 生成 k-SAT
def generate_k_sat(n_vars, n_clauses, k):
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = sample(range(1, n_vars + 1), k)
        clause = [var if randint(0, 1) else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses

# 调用 DeepSeek
def call_deepseek(cnf_text):
#     prompt = f"""You are a SAT logic solver.
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
# End your answer with: END.
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

            The formula is:
            {cnf_text}
            """

    response = openai.ChatCompletion.create(
        model=model_selected,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
        stop=["END."]
    )
    return response['choices'][0]['message']['content']

# 解析 DeepSeek 的返回
def parse_response(text):
    text = text.lower()
    sat = "unsatisfiable" not in text
    import re
    branches = re.search(r"branches.*?(\d+)", text)
    conflicts = re.search(r"conflicts.*?(\d+)", text)
    b = int(branches.group(1)) if branches else 0
    c = int(conflicts.group(1)) if conflicts else 0
    return sat, b, c

# 主循环
for alpha in tqdm(alpha_values, desc="Processing alpha values"):
    L = int(alpha * N)
    branches_list = []
    sat_count = 0

    for inst_idx in range(instances_per_alpha):
        cnf = generate_k_sat(N, L, k)
        cnf_text = cnf_to_prompt(cnf)

        try:
            response = call_deepseek(cnf_text)
            sat, branches, conflicts = parse_response(response)
        except Exception as e:
            print("API call failed:", e)
            continue

        if sat:
            sat_count += 1
        branches_list.append(branches)

        # 保存结果文件
        fname = f"cnf_k{k}_N{N}_L{L}_alpha{round(alpha,2)}_inst{inst_idx+1}.txt"
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write("c Random 3-SAT\n")
            f.write(f"c alpha={round(alpha,2)}, N={N}, L={L}, instance={inst_idx+1}\n")
            f.write("p cnf {} {}\n".format(N, L))
            for clause in cnf:
                f.write(" ".join(str(x) for x in clause) + " 0\n")
            f.write("\n")
            f.write(response.strip())
    print("branches_list:", branches_list)
    mean_branches.append(mean(branches_list))
    median_branches.append(median(branches_list))
    prob_sat.append(sat_count / len(branches_list))

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(alpha_values, mean_branches, label="Mean branches", color="black")
ax1.plot(alpha_values, median_branches, '--', label="Median branches", color="black")
ax1.set_xlabel("L / N")
ax1.set_ylabel("Number of branches")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(alpha_values, prob_sat, ':', label="Prob(sat)", color="black")
ax2.set_ylabel("Prob(sat)")

plt.title("DeepSeek on Random 3-SAT, N=75")
plt.grid(True)
plt.tight_layout()
plt.show()
