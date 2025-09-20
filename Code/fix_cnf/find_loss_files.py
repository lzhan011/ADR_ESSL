import os

if __name__ == '__main__':
    c_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_openai_prediction_o1'

    selected_files_list = os.listdir(c_root)
    missed_files_list = []
    for file in os.listdir(c_root):
        print(file)
        if "RC2_fixed" in file:
            unsat_file = file.replace('_RC2_fixed', '')
            if unsat_file not in selected_files_list:
                print("unsat_file:", unsat_file)
                missed_files_list.append(unsat_file)
        else:
            fixed_file = file[:-4]+"_RC2_fixed.cnf"
            if fixed_file not in selected_files_list:
                print("fixed_file:", fixed_file)
                missed_files_list.append(fixed_file)

    print(len(missed_files_list))
