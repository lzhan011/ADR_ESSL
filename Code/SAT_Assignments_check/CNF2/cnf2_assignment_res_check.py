import os.path

from pysat.solvers import Minisat22
import pandas as pd
import re
import json

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


import re, json

def _extract_json_from_line(line: str) -> dict:
    """从一行中提取 JSON 字符串并解析；容错处理 {{...}} 和 +123。"""
    start = line.find('{')
    end   = line.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in line.")

    obj = line[start:end+1]

    # 剥掉成对的外层 {{ ... }}（可能多层）
    while obj.startswith('{{') and obj.endswith('}}'):
        obj = obj[1:-1]

    # JSON 不允许 +数字：仅在 [: 或空白 或 [ 之后的 +n 替换，避免误伤字符串
    obj = re.sub(r'(?<=\[|\s|:)\+(\d+)', r'\1', obj)

    return json.loads(obj)

def parse_cnf_and_model(filepath: str):
    """
    从文件里解析：
      - clauses: List[List[int]]
      - model:   List[int]   （如果找到 JSON 且 answer=SATISFIABLE 且给出 assignment）
    """
    clauses = []
    model   = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ===== 解析 p 行，获取 N, L（可用于 sanity check）=====
    N_from_p = None
    L_from_p = None
    for line in lines:
        if line.startswith('p cnf'):
            parts = line.strip().split()
            # p cnf N L
            if len(parts) >= 4:
                try:
                    N_from_p = int(parts[2])
                    L_from_p = int(parts[3])
                except:
                    pass
            break

    # ===== 解析子句 =====
    # 规则：在出现 "p cnf" 后，连续若干行形如 "a b c ... 0" 的都是子句
    read_clause = False
    for line in lines:
        if line.startswith('p cnf'):
            read_clause = True
            continue

        if read_clause:
            s = line.strip()
            # 空行或非数字行 => 子句段落结束
            if s == "" or not any(ch.isdigit() for ch in s):
                read_clause = False
                continue

            # 提取整数并丢弃末尾 0
            parts = s.split()
            if parts and parts[-1] == '0':
                try:
                    clause = [int(x) for x in parts[:-1]]
                except:
                    clause = []
                if clause:
                    clauses.append(clause)
            else:
                # 不是规范子句行，结束子句段
                read_clause = False

    # ===== 在全文件范围里寻找 JSON 结果行（通常在末尾）=====
    assignment = None
    for line in lines:
        if '{' in line and '}' in line:
            try:
                pred = _extract_json_from_line(line)
                # 只要找到第一个含 "answer" 和 "assignment" 的就用它
                if isinstance(pred, dict) and "answer" in pred and "assignment" in pred:
                    ans = str(pred.get("answer", "")).upper()
                    if ans == "SATISFIABLE":
                        # 直接采用给定 assignment
                        raw_assign = pred.get("assignment", [])
                        try:
                            assignment = [int(x) for x in raw_assign]
                        except:
                            assignment = None
                    else:
                        assignment = []  # UNSAT 或未给出
                    break
            except Exception:
                continue

    # ===== 生成 model 返回 =====
    if assignment is not None:
        model = assignment
    else:
        model = []

    return clauses, model


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


def check_model_with_minisat(clauses, model, N):
    # 如果长度不对或者有 0，直接 False
    if N is not None and len(model) != N:
        return False
    if any(x == 0 for x in model):
        return False

    try:
        with Minisat22(bootstrap_with=clauses) as solver:
            return solver.solve(assumptions=model)
    except Exception:
        return False


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
        if model in file_dir:
            model_name = model
            break
    return model_name


def parse_one_model(model_name, combine_Separated_Correctly_and_Assignments_satisfied_rate):
    """
    返回:
      - res: 当前 model 的 (按目录) 汇总 DataFrame（原有）
      - file_level_df: 当前 model 的文件级结果 DataFrame（新增）
    """
    if combine_Separated_Correctly_and_Assignments_satisfied_rate:
        Separated_Correctly_dict = read_Separated_Correctly(model_list)

    res = []
    file_level_rows = []   # ⭐ 新增：每个文件的行

    for sub_dir in os.listdir(file_dir_parent):
        one_res = {}
        print("sub_dir:", sub_dir)

        if sub_dir.endswith('o1'):
            if not sub_dir.endswith('o1_openai_prediction_o1'):
                continue

        if not sub_dir.endswith(model_name):
            continue

        file_dir = os.path.join(file_dir_parent, sub_dir)

        if 'analysis' not in sub_dir:
            model_name_detected = get_model_name(model_list, file_dir)
            if combine_Separated_Correctly_and_Assignments_satisfied_rate:
                Separated_Correctly_one_model = Separated_Correctly_dict[model_name_detected]

            satisfied_number = 0
            not_satisfied_number = 0
            literals_number = None  # 避免未赋值引用

            for file_name in os.listdir(file_dir):
                literals_number = re.findall("N\d+", file_name)[0]
                literals_number = int(literals_number[1:])
                if combine_Separated_Correctly_and_Assignments_satisfied_rate:
                    Separated_Correctly_one_model_by_N = Separated_Correctly_one_model[
                        Separated_Correctly_one_model['N'] == literals_number]

                file_name_normalized = file_name.replace("_RC2_fixed", "")[:-4]

                if combine_Separated_Correctly_and_Assignments_satisfied_rate:
                    if file_name_normalized not in Separated_Correctly_one_model_by_N['Separated_Correctly_file_name'].to_list():
                        continue

                cnf_file_path = os.path.join(file_dir, file_name)
                print("cnf_file_path:", cnf_file_path)

                # 解析 CNF & assignment
                clauses, model = parse_cnf_and_model(cnf_file_path)

                # 默认 None：表示 assignment 不完整或异常
                is_satisfied = None
                if len(model) == int(literals_number):
                    print("model:", model)
                    print("literals_number:", literals_number)
                    ok = check_model_with_minisat(clauses, model, literals_number)
                    is_satisfied = bool(ok)
                    if ok:
                        satisfied_number += 1
                    else:
                        not_satisfied_number += 1

                # ⭐ 累计文件级记录
                file_level_rows.append({
                    "model_name": model_name_detected,
                    "N": literals_number,
                    "sub_dir": sub_dir,
                    "file_name": file_name,
                    "is_satisfied": is_satisfied
                })

            print("literals_number:", literals_number, "satisfied_number: ", satisfied_number,
                  "--- not_satisfied_number:", not_satisfied_number)
            one_res['sub_dir'] = sub_dir
            N = re.findall(r"N_(\d+)", sub_dir)[0]
            one_res["N"] = int(N)
            one_res['model_name'] = model_name_detected
            one_res['literals_number'] = literals_number
            one_res['Assignments_satisfied_number'] = satisfied_number
            one_res['Assignments_not_satisfied_number'] = not_satisfied_number
            sum_number = satisfied_number + not_satisfied_number
            if sum_number == 0:
                one_res['Assignments_satisfied_rate'] = 0
            else:
                one_res['Assignments_satisfied_rate'] = satisfied_number / sum_number
            res.append(one_res)

    # 原有汇总
    res = pd.DataFrame(res)
    if not res.empty:
        res = res.sort_values(by=['model_name', 'N'], ascending=[True, True])

    # 文件级结果 DataFrame（新增）
    file_level_df = pd.DataFrame(file_level_rows)
    if not file_level_df.empty:
        file_level_df = file_level_df.sort_values(by=['model_name', 'N', 'file_name'])

    # 将原有 csv 输出仍然保留
    if combine_Separated_Correctly_and_Assignments_satisfied_rate:
        res.to_csv(
            os.path.join(file_dir_parent, 'analysis', f'Assignments_satisfied_rate_combine_Separated_Correctly_and_{model_name}.csv'),
            index=False)
    else:
        res.to_csv(os.path.join(file_dir_parent, 'analysis', f'Assignments_satisfied_rate_{model_name}.csv'), index=False)

    return res, file_level_df


# =============== 主程序 ===============
file_dir_parent = r'/scratch/lzhan011/Satisfiability_Solvers/Code/CNF2/generate/cnf_results_CDCL/prediction_result'
model_list = ['gpt-5','o3-mini', 'gpt-4-turbo', 'gpt-4o', 'gpt-4.1', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'chatgpt-4o-latest', 'deepseek-reasoner',
              "claude-opus-4-20250514","claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219",  "claude-3-opus-20240229"]
# model_list = ['chatgpt-4o-latest']

combine_Separated_Correctly_and_Assignments_satisfied_rate = False

all_model_res = []
all_model_file_level = []   # ⭐ 新增：收集所有模型的文件级结果

for model_name in model_list:
    one_model_res, one_model_files = parse_one_model(model_name, combine_Separated_Correctly_and_Assignments_satisfied_rate)
    all_model_res.append(one_model_res)
    all_model_file_level.append(one_model_files)

# 汇总两个层级的结果
all_model_res = pd.concat(all_model_res, ignore_index=True) if len(all_model_res) else pd.DataFrame()
all_model_file_level = pd.concat(all_model_file_level, ignore_index=True) if len(all_model_file_level) else pd.DataFrame()

# 原有：写总的模型级汇总
if not combine_Separated_Correctly_and_Assignments_satisfied_rate:
    os.makedirs(os.path.join(file_dir_parent, 'analysis'), exist_ok=True)
    all_model_res.to_excel(os.path.join(file_dir_parent, 'analysis', 'three_ways_evaluation', "All_Model_cnf2_Assignments_satisfied_rate.xlsx"),
        index=False)

# 新增：写每文件级结果（一个文件，包含所有模型、所有 N）
if not all_model_file_level.empty:
    out_per_file = os.path.join(file_dir_parent, 'analysis', 'three_ways_evaluation', "All_Model_cnf2_Assigned_value_satisfied_per_file.xlsx")
    all_model_file_level.to_excel(out_per_file, index=False)
    print("保存文件级结果：", out_per_file)

print(all_model_res)
