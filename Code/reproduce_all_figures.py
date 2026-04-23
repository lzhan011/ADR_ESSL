#!/usr/bin/env python3

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(description, relative_path):
    print("========================================")
    print(f"Running {description}...")
    print("========================================")

    script_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(script_path):
        print(f"[ERROR] File not found: {script_path}")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Error while running {description}")
        sys.exit(1)

def main():
    print("========================================")
    print("Reproducing all figures...")
    print("========================================")

    run_step(
        "Figure 1(a)",
        "invoke_traditional_methond/phase_transition/phase_transition_generate_and_draw_Minisat22_only_draw_median.py"
    )

    run_step(
        "Figure 1(b)",
        "SAT_Draw_Figure/draw_curve_unsat.py"
    )

    run_step(
        "Figure 2",
        "invoke_openai/merge_two_kinds_of_dataset_plot_from_intermediate_file.py"
    )

    run_step(
        "Figure 3",
        "SAT_Draw_Figure/draw_curve_pairs.py"
    )

    run_step(
        "Figure 4",
        "SAT_Draw_Figure/draw_curve_2sat.py"
    )

    run_step(
        "Appendix Figure 1",
        "SAT_Draw_Figure/draw_curve_vertex_cover.py"
    )

    run_step(
        "Appendix Figure 2",
        "SAT_Draw_Figure/draw_3d_packing.py"
    )

    print("========================================")
    print("All figures have been generated.")
    print("========================================")

if __name__ == "__main__":
    main()