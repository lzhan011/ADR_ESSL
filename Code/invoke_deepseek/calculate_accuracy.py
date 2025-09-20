import os
import re
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行（如需要）
pd.set_option('display.max_rows', None)

# 设置列内容不省略
pd.set_option('display.max_colwidth', None)

# 禁用科学计数法（可选）
pd.set_option('display.float_format', '{:.6f}'.format)



# ---------- 工具函数 ----------
def extract_info_from_file(filepath: str):
    # print("filepath:", filepath)

    """
    解析单个结果文件，返回
      time, branches, conflicts, gpt_sat(bool/None), clauses(list[list[int]])
    —— 你原来的 extract_info_from_file() 函数直接拿来即可。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses, read_clause = [], False
    time_val, branches, conflicts = 0.0, 0, 0
    sat = None    # True / False / None
    sat_index = None
    branches_number, conflicts_number = None, None
    for line_index in range(len(lines)):
        line = lines[line_index]
        if line.startswith("p cnf"):
            read_clause = True
            continue
        if read_clause:
            if line.strip() == "" or not any(c.isdigit() for c in line):
                read_clause = False
                continue
            clause = [int(x) for x in line.strip().split() if x != "0"]
            clauses.append(clause)


        # 使用正则提取所有英文字母
        letters = re.findall(r'[a-zA-Z]', line)

        # 拼接成一个单词
        letters_word = ''.join(letters).lower()
        if letters_word == "unsatisfiable":
            sat = False
            sat_index = line_index
        elif letters_word == "satisfiable":
            sat = True
            sat_index = line_index

    if sat_index is not None:
        sat_index_next = sat_index + 1
        sat_index_next = lines[sat_index_next].strip()
        if len(sat_index_next) == 0:
            sat_index = sat_index + 1

        try:
            branches_index = sat_index + 1
            branches_number = lines[branches_index]
            branches_number = re.findall(r'\d*', branches_number)
            branches_number = int(''.join(branches_number))

            conflicts_index = sat_index + 2
            conflicts_number = lines[conflicts_index]
            conflicts_number = re.findall(r'\d*', conflicts_number)
            conflicts_number = int(''.join(conflicts_number))
        except:
            print("please check the file:", filepath)




    return time_val, branches_number, conflicts_number, sat, clauses



if __name__ == '__main__':

    correct_rate_list  = []
    model_select = "deepseek-chat"  # deepseek-reasoner
    root_dir = r'C:\Research\Vulnerability\Satisfiability_Solvers\Code\invoke_deepseek'
    for dir in os.listdir(root_dir):

        dir_path = os.path.join(root_dir, dir)

        # if dir_path.endswith("openai_prediction"):
        if dir_path.endswith(model_select):

            print("\n\ndir:", dir)
            unsat_cnt = 0
            file_num = 0

            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                time_val, branches_number, conflicts_number, sat, clauses = extract_info_from_file(file_path)
                if sat is not None:
                    file_num += 1
                    if sat == False:
                        unsat_cnt += 1

            correct_rate = unsat_cnt / file_num
            print("correct_rate:", correct_rate, "unsat_cnt:", unsat_cnt, "file_num:", file_num)

            dir_n = re.findall('\d+', dir)[0]
            correct_rate_list.append({ "dir": dir,
                                        "N": int(dir_n),
                                       "correct_rate:": correct_rate,
                                      "unsat_cnt:": unsat_cnt,
                                      "file_num:": file_num})
            correct_rate_list.sort(key=lambda x: x["N"])
    correct_rate_list = pd.DataFrame(correct_rate_list)
    correct_rate_list.to_excel(os.path.join(root_dir, "correct_rate_cross_n_"+str(model_select)+".xlsx"))
    # correct_rate_list.to_excel(os.path.join(root_dir, "correct_rate_cross_n_GPT-4.1_V2.xlsx"))

    print(correct_rate_list)





