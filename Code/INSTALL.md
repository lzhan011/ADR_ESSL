# INSTALL

## Artifact title
**Evaluating Satisfiability Solving with LLMs**

## Recommended installation and usage path

The recommended path for artifact evaluation is to reproduce the figures directly from the provided intermediate files and processed outputs. This avoids the need to rerun all commercial-model inference.

---

## 1. Create a Python environment

A virtual environment is recommended.

### Option A: `venv`

```bash
python -m venv artifact_env
source artifact_env/bin/activate
````

On Windows PowerShell:

```powershell
python -m venv artifact_env
artifact_env\Scripts\Activate.ps1
```

### Option B: conda

```bash
conda create -n artifact_env python=3.10 -y
conda activate artifact_env
```

---

## 2. Install required packages

Install the Python dependencies with:

```bash
pip install scikit-learn, pandas numpy matplotlib openpyxl openai anthropic requests python-sat tqdm
```

These packages are sufficient for the default reproduction workflow described in the README.

---

## 3. Basic installation check

After installation, verify that Python can import the main dependencies:

```bash
python -c "import scikit-learn, pandas, numpy, matplotlib, openpyxl, requests, tqdm; print('basic imports OK')"
```

Expected output:

```text
basic imports OK
```

If you also want to verify the SAT-solver package:

```bash
python -c "from pysat.solvers import Minisat22; print('python-sat OK')"
```

Expected output:

```text
python-sat OK
```

If these commands run successfully, the environment is ready for the default artifact workflow.

---

## 4. Recommended quick test

The fastest way to verify that the artifact is installed correctly and that the main empirical results can be reproduced is:

```bash
python reproduce_all_figures.py
```

This is the recommended artifact-evaluation entry point.

This script regenerates the paper’s main visual outputs from the prepared intermediate files included in the artifact.

---

## 5. Expected outputs from the quick test

A successful run should regenerate figure files under the output directories used by the paper. Representative expected outputs include:

* `Analysis_Result_Collection/Figure_in_paper/phase_transition/figures_CDCL_phase_transition/Random_3-SAT_CDCL_N_75_median.png`
* `Analysis_Result_Collection/Figure_in_paper/unsat/unsat_small_alpha_prediction_correct_rate.png`
* `Analysis_Result_Collection/Figure_in_paper/LLMs_phase_transition/combined_metrics/combined_metrics_4x3.png`
* `Analysis_Result_Collection/Figure_in_paper/pairs/pairs_small_alpha_prediction_correct_rate_3x3_only.png`
* `Analysis_Result_Collection/Figure_in_paper/2SAT/2SAT_3x3_only.png`
* `Analysis_Result_Collection/Figure_in_paper/Vertex_Cover/Vertex_Cover_legend_in_last_panel.png`
* `Analysis_Result_Collection/Figure_in_paper/3d_packing/packing_pred_llm_yes_3x3.png`

The exact file timestamps may differ depending on the local machine and whether some outputs already exist.

---

## 6. Optional figure-by-figure reproduction

If desired, individual figures can also be reproduced using their corresponding scripts.

### Figure 1(a): CDCL phase-transition baseline

```bash
python Code/invoke_traditional_methond/phase_transition/phase_transition_generate_and_draw_Minisat22_only_draw_median.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/phase_transition/figures_CDCL_phase_transition/Random_3-SAT_CDCL_N_75_median.png`

### Figure 1(b): low-α UNSAT detection

```bash
python Code/SAT_Draw_Figure/draw_curve_unsat.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/unsat/unsat_small_alpha_prediction_correct_rate.png`

### Figure 3: paired 3-SAT evaluation with ADR

```bash
python Code/SAT_Draw_Figure/draw_curve_pairs.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/pairs/pairs_small_alpha_prediction_correct_rate_3x3_only.png`

### Figure 4: paired 2-SAT evaluation

```bash
python Code/SAT_Draw_Figure/draw_curve_2sat.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/2SAT/2SAT_3x3_only.png`

### Appendix Figure 1: Vertex Cover

```bash
python Code/SAT_Draw_Figure/draw_curve_vertex_cover.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/Vertex_Cover/Vertex_Cover_legend_in_last_panel.png`

### Appendix Figure 2: 3D packing

```bash
python Code/SAT_Draw_Figure/draw_3d_packing.py
```

Expected representative output:

* `Analysis_Result_Collection/Figure_in_paper/3d_packing/packing_pred_llm_yes_3x3.png`

---

## 7. Optional commercial-model reruns

The artifact also contains scripts for rerunning commercial-model inference. These are **not required** for the recommended artifact check.

To use these scripts, set the corresponding API keys first:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Examples of such workflows include:

* OpenAI-based low-α UNSAT prediction
* DeepSeek-based low-α UNSAT prediction
* Anthropic-based low-α UNSAT prediction
* paired 3-SAT prediction
* 2-SAT prediction
* Vertex Cover prediction
* 3D packing prediction

Because these workflows depend on closed-source commercial services, they may require paid access and may not exactly match the original runtime environment if provider-side APIs or model versions have changed.

---

## 8. Notes on the intended evaluation mode

This artifact is intentionally organized so that evaluators can verify the paper’s main empirical findings without having to rerun all paid model calls.

The default and recommended evaluation path is therefore:

1. install the dependencies,
2. run the basic import checks,
3. run:

```bash
python reproduce_all_figures.py
```

4. confirm that the expected output figures are generated in the indicated directories.

---

## 9. Troubleshooting

### Missing package error

If Python reports that a package is missing, reinstall the dependencies:

```bash
pip install pandas numpy matplotlib openpyxl openai anthropic requests python-sat tqdm
```

### `pysat` / `python-sat` import issue

The package is installed via `python-sat`, but the import path used in code is typically:

```python
from pysat.solvers import Minisat22
```

### API-related error

If a script that invokes a commercial LLM fails, check:

* whether the corresponding API key is set,
* whether the API service is reachable,
* whether the provider account has valid paid access.

### No need to rerun paid APIs for the main check

If the goal is artifact evaluation rather than complete API reruns, use the included intermediate files and run:

```bash
python reproduce_all_figures.py
```

---

## 10. Successful installation criterion

The installation is considered successful if:

* the dependency import checks run without error, and
* `python reproduce_all_figures.py` completes successfully and regenerates the expected figure outputs in the repository.


