import json
import os
import pandas as pd


def read_3_level_similar_pairs_predictions(model, three_level_prediction_dir):
    three_level_predictions = {}
    for sim_level in ['high', 'medium', 'low']:
        three_level_file_name = f"{model}_{sim_level}_pairs_prediction_res.json"
        three_level_file_path = os.path.join(three_level_prediction_dir, three_level_file_name)
        with open(three_level_file_path, "r", encoding='utf-8') as f:
            one_level_predictions = json.load(f)
            one_level_predictions_new = {}
            for N, one_N_res in one_level_predictions.items():
                one_N_res_new = {}
                for file_name, v in one_N_res.items():
                    file_name_sim_level_index = file_name.index(sim_level)
                    file_name_match_with_original = file_name[:file_name_sim_level_index - 1]
                    if v != {}:
                        v['file_name']= file_name
                    one_N_res_new[file_name_match_with_original] = v
                one_level_predictions_new[N] = one_N_res_new
            three_level_predictions[sim_level] = one_level_predictions_new

    return three_level_predictions


def decide_Separated_Correct(pair_prediction):
    if pair_prediction['Predictions_after_fix'] == True and pair_prediction['Predictions_before_fix'] == False:
        return True
    else:
        return False




def calculate_group_ADR(one_pair_res):
    # for one_pair_res in pair_prediction_N:
    Original_Separated_Correct = one_pair_res.get('Original_Separated_Correct', False)
    high_sim_level_Separated_Correct = one_pair_res.get('high_sim_level_Separated_Correct', False)
    medium_sim_level_Separated_Correct = one_pair_res.get('medium_sim_level_Separated_Correct', False)
    low_sim_level_Separated_Correct = one_pair_res.get('low_sim_level_Separated_Correct', False)
    Separated_Correct_Count = Original_Separated_Correct+high_sim_level_Separated_Correct + medium_sim_level_Separated_Correct + low_sim_level_Separated_Correct
    one_pair_res['Separated_Correct_Count'] = Separated_Correct_Count


    no_prediction_count = 0
    with_prediction_res_pair_count = 1
    for sim_level in ['high', 'medium', 'low']:
        key_name = f"{sim_level}_sim_level_Separated_Correct"
        if key_name not in one_pair_res:
            no_prediction_count +=1
        else:
            with_prediction_res_pair_count += 1

    one_pair_res['no_prediction_res_pair_count']  = no_prediction_count
    one_pair_res['with_prediction_res_pair_count'] = with_prediction_res_pair_count


    return one_pair_res






def cross_distribution(df, output_file):
    import pandas as pd

    # 假设你的 DataFrame 叫 df
    # 两列分别是：with_prediction_res_pair_count 和 Separated_Correct_Count

    # 指定完整的类别范围（0–4）
    all_values = [0, 1, 2, 3, 4]

    # 1. 生成交叉表（频数表），并补全缺失列
    count_table = pd.crosstab(
        df['with_prediction_res_pair_count'],
        df['Separated_Correct_Count']
    ).reindex(columns=all_values, fill_value=0)

    # 2. 添加行合计
    count_table['Total'] = count_table.sum(axis=1)

    # 3. 生成百分比表（逐行除以 Total）
    percent_table = count_table.div(count_table['Total'], axis=0).round(3)

    # 4. 导出到 Excel（两个 sheet）
    # output_file = "pair_count_summary_full.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        count_table.to_excel(writer, sheet_name="Counts")
        percent_table.to_excel(writer, sheet_name="Percentages")

    print(f"频数表和百分比表已保存到 {output_file}")


if __name__ == '__main__':
    original_file_prediction = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N/analysis/pairs_prediction_res'
    three_level_prediction_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result/analysis/pairs_prediction_res'

    model_list = ['gpt-3.5-turbo', 'o1', 'gpt-5', 'deepseek-reasoner', 'gpt-3.5-turbo', 'gpt-4.1', 'gpt-4o', 'gpt-4-turbo', ] # ' o3-mini'
    # model_list = ['gpt-5']
    for model in model_list:
        original_file_name = f"{model}_original_pairs_prediction_res.json"
        original_file_path = os.path.join(original_file_prediction, original_file_name)
        json_data = {}
        with open(original_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            print(json_data)

        three_level_predictions_res = read_3_level_similar_pairs_predictions(model, three_level_prediction_dir)
        N_list = sorted(list(json_data.keys()))
        N_list = ['25']
        for N in N_list:
            pair_prediction_N = []
            one_N_pairs_prediction = json_data[N]
            for file_name, pair_prediction in one_N_pairs_prediction.items():
                if pair_prediction != {}:
                    print(file_name, pair_prediction)
                    pair_prediction['file_name'] = file_name
                    Separated_correct_flag = decide_Separated_Correct(pair_prediction)
                    pair_prediction['Original_Predictions_after_fix'] = pair_prediction['Predictions_after_fix']
                    pair_prediction['Original_Predictions_before_fix'] = pair_prediction['Predictions_before_fix']
                    pair_prediction['Original_Separated_Correct'] = Separated_correct_flag
                    #

                    for sim_level in ['high', 'medium', 'low']:
                        one_level_res = three_level_predictions_res[sim_level]
                        file_name_corresponding_res = one_level_res.get(N, {}).get(file_name, {})
                        if file_name_corresponding_res == {}:
                            print("miss file_name:", file_name)
                        else:
                            print(file_name_corresponding_res)
                            Separated_correct_flag = decide_Separated_Correct(file_name_corresponding_res)
                            pair_prediction[f'{sim_level}_sim_level_Predictions_after_fix'] = file_name_corresponding_res['Predictions_after_fix']
                            pair_prediction[f'{sim_level}_sim_level_Predictions_before_fix'] = file_name_corresponding_res['Predictions_before_fix']
                            pair_prediction[f'{sim_level}_sim_level_Separated_Correct'] = Separated_correct_flag
                    pair_prediction = calculate_group_ADR(pair_prediction)
                    pair_prediction_N.append(pair_prediction)
            pair_prediction_N = pd.DataFrame(pair_prediction_N)
            output_file = os.path.join(original_file_prediction, 'group_analysis', f"group_analysis_{model}_{N}_pairs_prediction_res.xlsx")
            pair_prediction_N.to_excel(output_file, sheet_name='raw_result')
            cross_distribution(pair_prediction_N, output_file)








