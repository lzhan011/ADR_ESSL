import os
from irt_core import run_irt_from_all_pairs, fit_all_via_cli



if __name__ == "__main__":
    root_dir = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    three_level_similar = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/similar_cnf_pairs/fixed_set_mul_N_similar_version_3/prediction_result'
    three_level_similar_analysis_dir = os.path.join(three_level_similar, 'analysis')
    four_level_similar_analysis_dir = os.path.join(three_level_similar_analysis_dir, 'three_ways_evaluation')

    # =====================
    # Edit these paths/params before running
    # =====================
    ALL_PAIRS_PATH = os.path.join(four_level_similar_analysis_dir, "pair_predictions_all_models.xlsx")  # <-- 修改为你真实的 all_pairs 路径
     # <-- 修改为希望输出的目录
    SEED = 0
    FIT_MODELS = ("2PL", "3PL", "4PL")  # 可改为 ("2PL",) 或 ("2PL","3PL") 等
    pair_or_instance_level = 'instance' # instance  pair
    OUT_DIR = os.path.join(three_level_similar_analysis_dir, r"irt_outputs", pair_or_instance_level)
    # 确保输出目录存在
    os.makedirs(OUT_DIR, exist_ok=True)

    import pandas as pd

    # 运行端到端流程：读取 all_pairs → ADR-Strict 计分 → (可用则)拟合 2PL/3PL/4PL → 导出
    artifacts = run_irt_from_all_pairs(
        all_pairs_path=ALL_PAIRS_PATH,
        out_dir=OUT_DIR,
        seed=SEED,
        fit_models=FIT_MODELS,
        pair_or_instance_level=pair_or_instance_level
    )

    # 打印关键产物路径
    for k, v in artifacts.items():
        print(f"{k}: {v}")

    # 一键跑 2PL/3PL/4PL（按文件存在自动跳过缺项）
    cli_artifacts = fit_all_via_cli(OUT_DIR, which=("2pl","3pl","4pl"))
    print("CLI artifacts:", cli_artifacts)