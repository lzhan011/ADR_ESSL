import json
import os
import os.path
import pandas as pd
import re
from pysat.solvers import Minisat22

# ===================== 解析与校验工具函数 =====================

def parse_cnf_and_model_bak(filepath):
    clauses = []
    model_dict = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    read_clause = False
    for line in lines:
        if line.startswith("p cnf"):
            read_clause = True
            continue

        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)

    # ✅ 用正则表达式匹配所有形如 x1=0、x2=F、x3=T 的赋值
    assignment_pattern = re.compile(r"x(\d+)\s*=\s*([01TF])")

    for line in lines:
        matches = assignment_pattern.findall(line)
        for var_str, val_str in matches:
            var_id = int(var_str)
            is_true = val_str in ['1', 'T', 'True', 'true']
            model_dict[var_id] = var_id if is_true else -var_id

    model = [model_dict[k] for k in sorted(model_dict)]  # 保持顺序可读性
    return clauses, model


def parse_cnf_and_model(filepath, N=None):
    clauses = []
    model_dict = {}
    sat = None
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ===== 读取 CNF 子句 =====
    read_clause = False
    for line in lines:
        if line.startswith("p cnf"):
            read_clause = True
            continue

        if read_clause:
            # 子句段落在遇到空行或不含数字的行结束
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            # 提取子句中的整数（去掉末尾的 0）
            clause = [int(x) for x in line.strip().split() if x != "0"]
            if clause:  # 防御空子句行
                clauses.append(clause)

        letter = re.findall(r'[A-Za-z]+', line)
        if len(letter) == 1:
            letter = letter[0]
            if letter.upper() == 'UNSATISFIABLE':
                sat = 'UNSATISFIABLE'
            elif letter.upper() == 'SATISFIABLE':
                sat = 'SATISFIABLE'

    # ===== 解析模型（变量赋值）=====
    # 支持: x1=true / false / t / f / 1 / 0（大小写均可）
    pat = re.compile(r"x\s*(\d+)\s*=\s*(true|false|t|f|1|0)", re.IGNORECASE)

    def put(k, v_str):
        v = v_str.strip().lower()
        is_true = v in ('1', 't', 'true')
        model_dict[k] = k if is_true else -k

    # 1) 优先：若某行的 '=' 次数恰好等于 N，则以该行为准一次性解析
    picked_line = None
    if N is not None:
        for line in lines:
            if line.count('=') == N:
                picked_line = line
                break
    if picked_line is not None:
        for var_str, val_str in pat.findall(picked_line):
            put(int(var_str), val_str)
    else:
        # 2) 兜底：在全文件范围内抓取所有匹配
        for line in lines:
            for var_str, val_str in pat.findall(line):
                put(int(var_str), val_str)

    # 将模型转成按变量编号升序的列表（与你原逻辑一致）
    model = [model_dict[k] for k in sorted(model_dict)]
    return clauses, model, sat


def violated_clauses(clauses, model):
    assignment = {abs(lit): (lit > 0) for lit in model}
    unsatisfied = []
    for idx, clause in enumerate(clauses):
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, None)
            if val is not None:
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
        if not satisfied:
            unsatisfied.append((idx, clause))
    return unsatisfied, assignment


def check_model_with_minisat(clauses, model):
    with Minisat22(bootstrap_with=clauses) as solver:
        result = solver.solve(assumptions=model)
        return result


# ===================== 聚合/工具 =====================

def read_Separated_Correctly(model_list):
    analysis_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N/analysis'
    Separated_Correctly_dict = {}
    for one_model in model_list:
        file_name = one_model + "_instances_res_cross_n.xlsx"
        file_name_Separated_Correctly = file_name[:-5] + "_Separated_Correctly.xlsx"
        file_path_Separated_Correctly = os.path.join(analysis_dir, file_name_Separated_Correctly)
        file_df_Separated_Correctly = pd.read_excel(file_path_Separated_Correctly)
        Separated_Correctly_dict[one_model] = file_df_Separated_Correctly
    return Separated_Correctly_dict


def get_model_name(model_list, file_dir):
    model_name = ""
    for model in model_list:
        if file_dir.endswith(model):
            model_name = model
            break
    return model_name


# ===================== 主流程 =====================

file_dir_parent = r'/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
model_list = [
    'gpt-5', 'o3-mini', 'gpt-4-turbo', 'gpt-4o', 'gpt-4.1',
    'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'chatgpt-4o-latest',
    'deepseek-reasoner', 'o1'
]


# Separated_Correctly_dict = read_Separated_Correctly(model_list)

res = []
file_level_results = []   # ⭐ 新增：每个文件级别的 is_satisfied 记录

for sub_dir in os.listdir(file_dir_parent):
    one_res = {}
    print("sub_dir:", sub_dir)

    # o1 的特殊目录筛选
    if sub_dir.endswith('o1'):
        if not sub_dir.endswith('o1_openai_prediction_o1'):
            continue

    file_dir = os.path.join(file_dir_parent, sub_dir)
    model_name = get_model_name(model_list, file_dir)
    if model_name not in model_list:
        continue


    # 跳过 analysis 目录
    if 'analysis' not in sub_dir:

        satisfied_number = 0
        not_satisfied_number = 0
        literals_number = None

        for file_name in os.listdir(file_dir):
            # N 从文件名中提取
            literals_number = re.findall("N\d+", file_name)[0]
            literals_number = int(literals_number[1:])

            cnf_file_path = os.path.join(file_dir, file_name)
            clauses, model, sat = parse_cnf_and_model(cnf_file_path, N=literals_number)

            # 默认未计算（例如变量数不匹配时）
            is_satisfied = None

            # 满足才校验
            if len(model) == int(literals_number):
                is_satisfied = check_model_with_minisat(clauses, model)

                if not is_satisfied:
                    not_satisfied_number += 1
                else:
                    satisfied_number += 1

            # ⭐ 记录文件级别结果
            file_level_results.append({
                "sub_dir": sub_dir,
                "model_name": model_name,
                "N": literals_number,
                "file_name": file_name,
                "is_satisfied": (bool(is_satisfied) if is_satisfied is not None else None),
                "model_length_equals_N": (len(model) == int(literals_number)),
                "sat_flag_in_file": sat  # 可选：保留你在文件里解析到的 SAT/UNSAT 标记
            })

        print("literals_number:", literals_number,
              "satisfied_number: ", satisfied_number,
              "--- not_satisfied_number:", not_satisfied_number)

        one_res['sub_dir'] = sub_dir
        N = re.findall(r"N_(\d+)", sub_dir)[0]
        one_res["N"] = int(N)
        one_res['model_name'] = model_name
        one_res['literals_number'] = literals_number
        one_res['Assignments_satisfied_number'] = satisfied_number
        one_res['Assignments_not_satisfied_number'] = not_satisfied_number
        denom = (satisfied_number + not_satisfied_number)
        one_res['Assignments_satisfied_rate'] = (satisfied_number / denom) if denom else 0.0
        res.append(one_res)

# ===================== 保存结果 =====================

# 1) 汇总（原有）
res = pd.DataFrame(res)
res = res.sort_values(by=['model_name', 'N'], ascending=[True, True])
os.makedirs(os.path.join(file_dir_parent, 'analysis'), exist_ok=True)
res.to_csv(os.path.join(file_dir_parent, 'analysis', 'three_ways_evaluation', 'Assigned_value_satisfied_rate_all_models.csv'), index=False)

# 2) 文件级别结果（新增）
file_level_df = pd.DataFrame(file_level_results)
file_level_df = file_level_df.sort_values(by=['model_name', 'N', 'file_name'])
file_level_out = os.path.join(file_dir_parent, 'analysis', 'three_ways_evaluation', 'Assigned_value_satisfied_result_per_file.csv')
file_level_df.to_csv(file_level_out, index=False)

print("保存汇总：", os.path.join(file_dir_parent, 'analysis', 'three_ways_evaluation', 'Assigned_value_satisfied_rate_all_models.csv'))
print("保存文件级：", file_level_out)
