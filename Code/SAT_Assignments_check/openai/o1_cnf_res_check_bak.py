import os.path

from pysat.solvers import Minisat22
import pandas as pd
import re

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



import re

def parse_cnf_and_model(filepath, N=None):
    clauses = []
    model_dict = {}

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




def check_model_with_minisat(clauses, model):
    with Minisat22(bootstrap_with=clauses) as solver:
        result = solver.solve(assumptions=model)
        return result


# file_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_openai\draw_o1_phase_transition_figures\draw_o1_cnf_alpha_3_6_N_75_openai_prediction_o1'
# file_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set'
# file_dir = r'/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_openai_prediction_o1'





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




file_dir_parent = r'/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
model_list = ['o3-mini', 'gpt-4-turbo', 'gpt-4o', 'gpt-4.1', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'chatgpt-4o-latest', 'deepseek-reasoner', 'o1']



Separated_Correctly_dict = read_Separated_Correctly(model_list)



combine_Separated_Correctly_and_Assignments_satisfied_rate = False
model_name = ""
res = []
for sub_dir in os.listdir(file_dir_parent):
    one_res = {}
    print("sub_dir:", sub_dir)
    # if sub_dir.endswith('deepseek-reasoner') and '5' in sub_dir:
    #     print("sub_dir:", sub_dir)
    # else:
    #     continue

    if sub_dir.endswith('o1'):
        if not sub_dir.endswith('o1_openai_prediction_o1'):
            continue

    file_dir = os.path.join(file_dir_parent, sub_dir)

    if  'analysis' not in sub_dir:
        model_name = get_model_name(model_list, file_dir)
        Separated_Correctly_one_model = Separated_Correctly_dict[model_name]

        satisfied_number = 0
        not_satisfied_number = 0
        for file_name in  os.listdir(file_dir):

            # file_name = 'k3_N75_L225_alpha3.0_inst2_SAT.cnf'
            # file_name = "k3_N75_L412_alpha5.5_inst91_UNSAT.cnf"

            literals_number = re.findall("N\d+", file_name)[0]
            literals_number = int(literals_number[1:])

            Separated_Correctly_one_model_by_N = Separated_Correctly_one_model[Separated_Correctly_one_model['N'] == literals_number]

            file_name_normalized = file_name.replace("_RC2_fixed", "")[:-4]

            if combine_Separated_Correctly_and_Assignments_satisfied_rate:
                if file_name_normalized not in Separated_Correctly_one_model_by_N['Separated_Correctly_file_name'].to_list():
                    continue

            # 使用方法
            cnf_file_path = os.path.join(file_dir, file_name)  # 替换为你的路径
            # clauses, model = parse_cnf_and_model(cnf_file_path)
            clauses, model = parse_cnf_and_model(cnf_file_path, N=literals_number)
            if len(model) == int(literals_number):
                is_satisfied = check_model_with_minisat(clauses, model)
                # print("模型满足 CNF：" if is_satisfied else "模型不满足 CNF")


                if not is_satisfied:
                    not_satisfied_number += 1
                    violated, assignment = violated_clauses(clauses, model)
                    if violated:
                        pass
                        # print(f"有 {len(violated)} 个子句被违反：\n")
                        # for idx, clause in violated:
                        #     print(f"子句 {idx}: {clause}")
                        #     for lit in clause:
                        #         var = abs(lit)
                        #         var_value = assignment.get(var, "未赋值")
                        #         print(f"    x{var} = {var_value}")
                        #     print()
                else:
                    satisfied_number += 1

        print("literals_number:", literals_number, "satisfied_number: ", satisfied_number, "--- not_satisfied_number:", not_satisfied_number)
        one_res['sub_dir'] = sub_dir
        N = re.findall(r"N_(\d+)", sub_dir)[0]
        # for model in model_list:
        #     if model in file_dir:
        #         model_name = model
        #         break
        one_res["N"] = int(N)
        one_res['model_name']= model_name
        one_res['literals_number'] = literals_number
        one_res['Assignments_satisfied_number'] = satisfied_number
        one_res['Assignments_not_satisfied_number'] = not_satisfied_number
        sum_number = satisfied_number + not_satisfied_number
        if sum_number == 0:
            one_res['Assignments_satisfied_rate'] = 0
        else:
            one_res['Assignments_satisfied_rate'] = satisfied_number /   sum_number
        res.append(one_res)

res = pd.DataFrame(res)
res = res.sort_values(by=['model_name', 'N'], ascending=[True, True])
if combine_Separated_Correctly_and_Assignments_satisfied_rate:
    res.to_csv(os.path.join(file_dir_parent, 'analysis', 'combine_Separated_Correctly_and_Assignments_satisfied_rate.csv'), index=False)
else:
    res.to_csv(os.path.join(file_dir_parent, 'analysis', 'Assigned_value_satisfied_rate.csv'), index=False)

