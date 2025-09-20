import os

import pandas as pd
from collections import Counter



def check_prediction_is_correct(x):
    if x['label_IS_SAT'] == x['prediction_IS_SAT']:
        return True
    else:
        return False





if __name__ == '__main__':
    c_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N/analysis'
    model_list = ['o3-mini', 'gpt-4-turbo', 'gpt-4o', 'gpt-4.1', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo',
                  'chatgpt-4o-latest', 'deepseek-reasoner', 'o1']

    # model_list = ['o1']
    # model_list = ['deepseek-reasoner']
    for model in model_list:
        file_name = model + "_instances_res_cross_n.xlsx"
        res_df = pd.read_excel(os.path.join(c_root, file_name))
        file_name_list = res_df['file_name'].tolist()
        file_name_list = sorted(list([item.replace("_RC2_fixed", "")[:-4] for item in file_name_list]))
        # 统计每个 file_name 出现的次数
        counter = Counter(file_name_list)
        # 只保留那些出现两次的 file_name
        file_name_list_filtered = [name for name in file_name_list if counter[name] == 2]
        file_name_list_filtered = list(set(file_name_list_filtered))
        res_df['file_name_list_filtered'] = res_df['file_name'].apply(lambda x: x.replace("_RC2_fixed", "")[:-4])

        res_df['prediction_is_correct'] = res_df.apply(lambda x: check_prediction_is_correct(x), axis=1)

        res_df_filter = res_df[res_df['file_name_list_filtered'].isin(file_name_list_filtered)]

        Separated_correct_pair = []

        N_list = sorted(res_df['N'].unique())
        for one_N in N_list:
            for one_file_name in file_name_list_filtered:

                Number_Separated_correct_for_one_pair = 0

                res_df_N = res_df[res_df['N'] == one_N]
                res_df_N_selected_file = res_df_N[res_df_N['file_name_list_filtered'] == one_file_name]
                for index, row in res_df_N_selected_file.iterrows():
                    file_name_in_df = row['file_name']
                    N = row['N']
                    file_name_in_df = file_name_in_df.replace("_RC2_fixed", "")[:-4]
                    prediction_is_correct = row['prediction_is_correct']

                    if file_name_in_df == one_file_name and N == one_N:
                        if prediction_is_correct:
                            Number_Separated_correct_for_one_pair += 1
                            if Number_Separated_correct_for_one_pair == 2:
                                Separated_correct_pair.append({"N":N,
                                                               "Separated_Correctly_file_name": one_file_name})
                                break


        print("Separated_correct_pair:", Separated_correct_pair)
        print("Separated_correct_pair len:", Separated_correct_pair)

        Separated_correct_pair = pd.DataFrame(Separated_correct_pair)
        file_name_output = file_name[:-5] + "_Separated_Correctly.xlsx"
        Separated_correct_pair.to_excel(os.path.join(c_root, file_name_output))


# Separated_correct_pair: ['cnf_k3_N10_L35_alpha3.5_inst1057', 'cnf_k3_N10_L35_alpha3.5_inst1057', 'cnf_k3_N10_L35_alpha3.5_inst80', 'cnf_k3_N10_L35_alpha3.5_inst80', 'cnf_k3_N5_L17_alpha3.5_inst101', 'cnf_k3_N5_L17_alpha3.5_inst101', 'cnf_k3_N5_L17_alpha3.5_inst102', 'cnf_k3_N5_L17_alpha3.5_inst102', 'cnf_k3_N5_L17_alpha3.5_inst1034', 'cnf_k3_N5_L17_alpha3.5_inst1034', 'cnf_k3_N5_L17_alpha3.5_inst1069', 'cnf_k3_N5_L17_alpha3.5_inst1069', 'cnf_k3_N5_L17_alpha3.5_inst1078', 'cnf_k3_N5_L17_alpha3.5_inst1078', 'cnf_k3_N5_L17_alpha3.5_inst1084', 'cnf_k3_N5_L17_alpha3.5_inst1084', 'cnf_k3_N5_L17_alpha3.5_inst109', 'cnf_k3_N5_L17_alpha3.5_inst109', 'cnf_k3_N5_L17_alpha3.5_inst1092', 'cnf_k3_N5_L17_alpha3.5_inst1092', 'cnf_k3_N5_L17_alpha3.5_inst1149', 'cnf_k3_N5_L17_alpha3.5_inst1149', 'cnf_k3_N5_L17_alpha3.5_inst1155', 'cnf_k3_N5_L17_alpha3.5_inst1155', 'cnf_k3_N5_L17_alpha3.5_inst1169', 'cnf_k3_N5_L17_alpha3.5_inst1169', 'cnf_k3_N5_L17_alpha3.5_inst1220', 'cnf_k3_N5_L17_alpha3.5_inst1220', 'cnf_k3_N5_L17_alpha3.5_inst1233', 'cnf_k3_N5_L17_alpha3.5_inst1233', 'cnf_k3_N5_L17_alpha3.5_inst1251', 'cnf_k3_N5_L17_alpha3.5_inst1251', 'cnf_k3_N5_L17_alpha3.5_inst1265', 'cnf_k3_N5_L17_alpha3.5_inst1265', 'cnf_k3_N5_L17_alpha3.5_inst1277', 'cnf_k3_N5_L17_alpha3.5_inst1277', 'cnf_k3_N5_L17_alpha3.5_inst129', 'cnf_k3_N5_L17_alpha3.5_inst129', 'cnf_k3_N5_L17_alpha3.5_inst1321', 'cnf_k3_N5_L17_alpha3.5_inst1321', 'cnf_k3_N5_L17_alpha3.5_inst1322', 'cnf_k3_N5_L17_alpha3.5_inst1322', 'cnf_k3_N5_L17_alpha3.5_inst1346', 'cnf_k3_N5_L17_alpha3.5_inst1346', 'cnf_k3_N5_L17_alpha3.5_inst1416', 'cnf_k3_N5_L17_alpha3.5_inst1416', 'cnf_k3_N5_L17_alpha3.5_inst1488', 'cnf_k3_N5_L17_alpha3.5_inst1488', 'cnf_k3_N5_L17_alpha3.5_inst1491', 'cnf_k3_N5_L17_alpha3.5_inst1491', 'cnf_k3_N5_L17_alpha3.5_inst1492', 'cnf_k3_N5_L17_alpha3.5_inst1492', 'cnf_k3_N5_L17_alpha3.5_inst1498', 'cnf_k3_N5_L17_alpha3.5_inst1498', 'cnf_k3_N5_L17_alpha3.5_inst1522', 'cnf_k3_N5_L17_alpha3.5_inst1522', 'cnf_k3_N5_L17_alpha3.5_inst1530', 'cnf_k3_N5_L17_alpha3.5_inst1530', 'cnf_k3_N5_L17_alpha3.5_inst1556', 'cnf_k3_N5_L17_alpha3.5_inst1556', 'cnf_k3_N5_L17_alpha3.5_inst1578', 'cnf_k3_N5_L17_alpha3.5_inst1578', 'cnf_k3_N5_L17_alpha3.5_inst1595', 'cnf_k3_N5_L17_alpha3.5_inst1595', 'cnf_k3_N5_L17_alpha3.5_inst1619', 'cnf_k3_N5_L17_alpha3.5_inst1619', 'cnf_k3_N5_L17_alpha3.5_inst1644', 'cnf_k3_N5_L17_alpha3.5_inst1644', 'cnf_k3_N5_L17_alpha3.5_inst1673', 'cnf_k3_N5_L17_alpha3.5_inst1673', 'cnf_k3_N5_L17_alpha3.5_inst1716', 'cnf_k3_N5_L17_alpha3.5_inst1716', 'cnf_k3_N5_L17_alpha3.5_inst1734', 'cnf_k3_N5_L17_alpha3.5_inst1734', 'cnf_k3_N5_L17_alpha3.5_inst1757', 'cnf_k3_N5_L17_alpha3.5_inst1757', 'cnf_k3_N5_L17_alpha3.5_inst1768', 'cnf_k3_N5_L17_alpha3.5_inst1768', 'cnf_k3_N5_L17_alpha3.5_inst1772', 'cnf_k3_N5_L17_alpha3.5_inst1772', 'cnf_k3_N5_L17_alpha3.5_inst1778', 'cnf_k3_N5_L17_alpha3.5_inst1778', 'cnf_k3_N5_L17_alpha3.5_inst1797', 'cnf_k3_N5_L17_alpha3.5_inst1797', 'cnf_k3_N5_L17_alpha3.5_inst1849', 'cnf_k3_N5_L17_alpha3.5_inst1849', 'cnf_k3_N5_L17_alpha3.5_inst1862', 'cnf_k3_N5_L17_alpha3.5_inst1862', 'cnf_k3_N5_L17_alpha3.5_inst1877', 'cnf_k3_N5_L17_alpha3.5_inst1877', 'cnf_k3_N5_L17_alpha3.5_inst1898', 'cnf_k3_N5_L17_alpha3.5_inst1898', 'cnf_k3_N5_L17_alpha3.5_inst1904', 'cnf_k3_N5_L17_alpha3.5_inst1904', 'cnf_k3_N5_L17_alpha3.5_inst1909', 'cnf_k3_N5_L17_alpha3.5_inst1909', 'cnf_k3_N5_L17_alpha3.5_inst236', 'cnf_k3_N5_L17_alpha3.5_inst236', 'cnf_k3_N5_L17_alpha3.5_inst241', 'cnf_k3_N5_L17_alpha3.5_inst241', 'cnf_k3_N5_L17_alpha3.5_inst247', 'cnf_k3_N5_L17_alpha3.5_inst247', 'cnf_k3_N5_L17_alpha3.5_inst265', 'cnf_k3_N5_L17_alpha3.5_inst265', 'cnf_k3_N5_L17_alpha3.5_inst361', 'cnf_k3_N5_L17_alpha3.5_inst361', 'cnf_k3_N5_L17_alpha3.5_inst369', 'cnf_k3_N5_L17_alpha3.5_inst369', 'cnf_k3_N5_L17_alpha3.5_inst390', 'cnf_k3_N5_L17_alpha3.5_inst390', 'cnf_k3_N5_L17_alpha3.5_inst411', 'cnf_k3_N5_L17_alpha3.5_inst411', 'cnf_k3_N5_L17_alpha3.5_inst478', 'cnf_k3_N5_L17_alpha3.5_inst478', 'cnf_k3_N5_L17_alpha3.5_inst502', 'cnf_k3_N5_L17_alpha3.5_inst502', 'cnf_k3_N5_L17_alpha3.5_inst546', 'cnf_k3_N5_L17_alpha3.5_inst546', 'cnf_k3_N5_L17_alpha3.5_inst566', 'cnf_k3_N5_L17_alpha3.5_inst566', 'cnf_k3_N5_L17_alpha3.5_inst59', 'cnf_k3_N5_L17_alpha3.5_inst59', 'cnf_k3_N5_L17_alpha3.5_inst607', 'cnf_k3_N5_L17_alpha3.5_inst607', 'cnf_k3_N5_L17_alpha3.5_inst614', 'cnf_k3_N5_L17_alpha3.5_inst614', 'cnf_k3_N5_L17_alpha3.5_inst624', 'cnf_k3_N5_L17_alpha3.5_inst624', 'cnf_k3_N5_L17_alpha3.5_inst775', 'cnf_k3_N5_L17_alpha3.5_inst775', 'cnf_k3_N5_L17_alpha3.5_inst802', 'cnf_k3_N5_L17_alpha3.5_inst802', 'cnf_k3_N5_L17_alpha3.5_inst819', 'cnf_k3_N5_L17_alpha3.5_inst819', 'cnf_k3_N5_L17_alpha3.5_inst823', 'cnf_k3_N5_L17_alpha3.5_inst823', 'cnf_k3_N5_L17_alpha3.5_inst888', 'cnf_k3_N5_L17_alpha3.5_inst888', 'cnf_k3_N5_L17_alpha3.5_inst902', 'cnf_k3_N5_L17_alpha3.5_inst902', 'cnf_k3_N8_L28_alpha3.5_inst1052', 'cnf_k3_N8_L28_alpha3.5_inst1052', 'cnf_k3_N8_L28_alpha3.5_inst1067', 'cnf_k3_N8_L28_alpha3.5_inst1067', 'cnf_k3_N8_L28_alpha3.5_inst1074', 'cnf_k3_N8_L28_alpha3.5_inst1074', 'cnf_k3_N8_L28_alpha3.5_inst1087', 'cnf_k3_N8_L28_alpha3.5_inst1087', 'cnf_k3_N8_L28_alpha3.5_inst113', 'cnf_k3_N8_L28_alpha3.5_inst113', 'cnf_k3_N8_L28_alpha3.5_inst1175', 'cnf_k3_N8_L28_alpha3.5_inst1175', 'cnf_k3_N8_L28_alpha3.5_inst1177', 'cnf_k3_N8_L28_alpha3.5_inst1177', 'cnf_k3_N8_L28_alpha3.5_inst1180', 'cnf_k3_N8_L28_alpha3.5_inst1180', 'cnf_k3_N8_L28_alpha3.5_inst1199', 'cnf_k3_N8_L28_alpha3.5_inst1199', 'cnf_k3_N8_L28_alpha3.5_inst1212', 'cnf_k3_N8_L28_alpha3.5_inst1212', 'cnf_k3_N8_L28_alpha3.5_inst1243', 'cnf_k3_N8_L28_alpha3.5_inst1243', 'cnf_k3_N8_L28_alpha3.5_inst1247', 'cnf_k3_N8_L28_alpha3.5_inst1247', 'cnf_k3_N8_L28_alpha3.5_inst1252', 'cnf_k3_N8_L28_alpha3.5_inst1252', 'cnf_k3_N8_L28_alpha3.5_inst126', 'cnf_k3_N8_L28_alpha3.5_inst126', 'cnf_k3_N8_L28_alpha3.5_inst1261', 'cnf_k3_N8_L28_alpha3.5_inst1261', 'cnf_k3_N8_L28_alpha3.5_inst1363', 'cnf_k3_N8_L28_alpha3.5_inst1363', 'cnf_k3_N8_L28_alpha3.5_inst1431', 'cnf_k3_N8_L28_alpha3.5_inst1431', 'cnf_k3_N8_L28_alpha3.5_inst1446', 'cnf_k3_N8_L28_alpha3.5_inst1446', 'cnf_k3_N8_L28_alpha3.5_inst194', 'cnf_k3_N8_L28_alpha3.5_inst194', 'cnf_k3_N8_L28_alpha3.5_inst196', 'cnf_k3_N8_L28_alpha3.5_inst196', 'cnf_k3_N8_L28_alpha3.5_inst206', 'cnf_k3_N8_L28_alpha3.5_inst206', 'cnf_k3_N8_L28_alpha3.5_inst22', 'cnf_k3_N8_L28_alpha3.5_inst22', 'cnf_k3_N8_L28_alpha3.5_inst229', 'cnf_k3_N8_L28_alpha3.5_inst229', 'cnf_k3_N8_L28_alpha3.5_inst244', 'cnf_k3_N8_L28_alpha3.5_inst244', 'cnf_k3_N8_L28_alpha3.5_inst248', 'cnf_k3_N8_L28_alpha3.5_inst248', 'cnf_k3_N8_L28_alpha3.5_inst251', 'cnf_k3_N8_L28_alpha3.5_inst251', 'cnf_k3_N8_L28_alpha3.5_inst276', 'cnf_k3_N8_L28_alpha3.5_inst276', 'cnf_k3_N8_L28_alpha3.5_inst278', 'cnf_k3_N8_L28_alpha3.5_inst278', 'cnf_k3_N8_L28_alpha3.5_inst28', 'cnf_k3_N8_L28_alpha3.5_inst28', 'cnf_k3_N8_L28_alpha3.5_inst301', 'cnf_k3_N8_L28_alpha3.5_inst301', 'cnf_k3_N8_L28_alpha3.5_inst330', 'cnf_k3_N8_L28_alpha3.5_inst330', 'cnf_k3_N8_L28_alpha3.5_inst345', 'cnf_k3_N8_L28_alpha3.5_inst345', 'cnf_k3_N8_L28_alpha3.5_inst378', 'cnf_k3_N8_L28_alpha3.5_inst378', 'cnf_k3_N8_L28_alpha3.5_inst385', 'cnf_k3_N8_L28_alpha3.5_inst385', 'cnf_k3_N8_L28_alpha3.5_inst399', 'cnf_k3_N8_L28_alpha3.5_inst399', 'cnf_k3_N8_L28_alpha3.5_inst414', 'cnf_k3_N8_L28_alpha3.5_inst414', 'cnf_k3_N8_L28_alpha3.5_inst434', 'cnf_k3_N8_L28_alpha3.5_inst434', 'cnf_k3_N8_L28_alpha3.5_inst444', 'cnf_k3_N8_L28_alpha3.5_inst444', 'cnf_k3_N8_L28_alpha3.5_inst486', 'cnf_k3_N8_L28_alpha3.5_inst486', 'cnf_k3_N8_L28_alpha3.5_inst55', 'cnf_k3_N8_L28_alpha3.5_inst55', 'cnf_k3_N8_L28_alpha3.5_inst553', 'cnf_k3_N8_L28_alpha3.5_inst553', 'cnf_k3_N8_L28_alpha3.5_inst580', 'cnf_k3_N8_L28_alpha3.5_inst580', 'cnf_k3_N8_L28_alpha3.5_inst605', 'cnf_k3_N8_L28_alpha3.5_inst605', 'cnf_k3_N8_L28_alpha3.5_inst66', 'cnf_k3_N8_L28_alpha3.5_inst66', 'cnf_k3_N8_L28_alpha3.5_inst664', 'cnf_k3_N8_L28_alpha3.5_inst664', 'cnf_k3_N8_L28_alpha3.5_inst673', 'cnf_k3_N8_L28_alpha3.5_inst673', 'cnf_k3_N8_L28_alpha3.5_inst692', 'cnf_k3_N8_L28_alpha3.5_inst692', 'cnf_k3_N8_L28_alpha3.5_inst698', 'cnf_k3_N8_L28_alpha3.5_inst698', 'cnf_k3_N8_L28_alpha3.5_inst724', 'cnf_k3_N8_L28_alpha3.5_inst724', 'cnf_k3_N8_L28_alpha3.5_inst73', 'cnf_k3_N8_L28_alpha3.5_inst73', 'cnf_k3_N8_L28_alpha3.5_inst755', 'cnf_k3_N8_L28_alpha3.5_inst755', 'cnf_k3_N8_L28_alpha3.5_inst756', 'cnf_k3_N8_L28_alpha3.5_inst756', 'cnf_k3_N8_L28_alpha3.5_inst778', 'cnf_k3_N8_L28_alpha3.5_inst778', 'cnf_k3_N8_L28_alpha3.5_inst796', 'cnf_k3_N8_L28_alpha3.5_inst796', 'cnf_k3_N8_L28_alpha3.5_inst853', 'cnf_k3_N8_L28_alpha3.5_inst853', 'cnf_k3_N8_L28_alpha3.5_inst886', 'cnf_k3_N8_L28_alpha3.5_inst886', 'cnf_k3_N8_L28_alpha3.5_inst891', 'cnf_k3_N8_L28_alpha3.5_inst891', 'cnf_k3_N8_L28_alpha3.5_inst925', 'cnf_k3_N8_L28_alpha3.5_inst925']
