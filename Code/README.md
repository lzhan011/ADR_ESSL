

# Artifact README

## Evaluating Satisfiability Solving with LLMs

This repository contains the artifact for the paper:

**Evaluating Satisfiability Solving with LLMs**

The artifact is designed to support **reproducibility** for the paper. It provides implementations for:

* **CNF generation**
* **paired-CNF generation**
* **LLM API invocation** for obtaining model predictions
* **traditional evaluation metrics**
* the proposed **Accurate Differentiation Rate (ADR)**
* **cross-representation reductions** to **Vertex Cover** and **discrete 3D packing**
* **deterministic witness validation** for assignments, covers, and packing outputs

In addition, the artifact contains plotting scripts, aggregated intermediate files, selected prediction-result directories, prompts, and appendix-related supplementary materials needed to reproduce the main findings of the paper. The experiments cover:

* random 3-SAT phase-transition analysis,
* LLM performance on low-(\alpha) UNSAT CNFs,
* paired SAT/UNSAT evaluation with **ADR**,
* 2-SAT as a controlled baseline,
* cross-representation evaluations on **Vertex Cover** and **discrete 3D packing**.

---

# 1. Artifact Scope

This artifact supports two levels of reproducibility.

## 1.1 Fast reproduction from aggregated intermediate data

Most figures in the paper and appendix can be regenerated directly from prepared intermediate `.xlsx` files. This is the **recommended** path for artifact evaluation.

## 1.2 Reproduction from prediction-result directories

For several experiments, we also provide prediction-result directories and solver-result directories that were used to construct the aggregated summaries. This path provides greater provenance and traceability.

## 1.3 Partial end-to-end reruns

The repository also includes scripts for:

* generating SAT / UNSAT instances,
* generating paired instances,
* invoking commercial LLM APIs,
* parsing predictions,
* validating witnesses,
* recomputing metrics.


---

# 2. Quick Start

## 2.1 Recommended quick check

To regenerate the main figures directly from prepared intermediate files:

```bash
python reproduce_all_figures.py
```

This is the fastest way to verify the main empirical claims of the paper.

## 2.2 Minimal setup

Create a Python environment and install dependencies:

```bash
pip install pandas numpy matplotlib openpyxl openai anthropic requests python-sat tqdm
```

If your local environment already includes some of these packages, you may omit them as needed.

## 2.3 What this quick path reproduces

The quick path reproduces the paper’s visual results from prepared intermediate artifacts, including:

* the classical 3-SAT phase-transition baseline,
* low-(\alpha) UNSAT detection trends,
* paired-formula evaluation with ADR,
* 2-SAT paired evaluation,
* cross-representation plots for Vertex Cover and 3D packing.

---

# 3. Artifact Directory Structure

This section describes the repository structure following the experimental flow of the paper.

## 3.1 `invoke_traditional_methond`

This directory serves as the experimental baseline, corresponding to **Figure 1(a)**.

* **Functionality**: invokes a standard **CDCL** SAT solver on CNF instances.
* **Purpose**:

  * reproduces the classical 3-SAT **phase transition**,
  * measures satisfiability probability and search hardness,
  * provides the algorithmic baseline used to contrast LLM behavior.

## Figure 1(a)  Phase transition for random 3SAT with 𝑁 =75 using a CDCL solver.

1. **Figure 1(a) generation script**  
   The workflow for Figure 1(a), including input generation, CDCL conflict-count computation, and phase-transition plotting, is implemented in:

   ```text
   Code/invoke_traditional_methond/phase_transition/phase_transition_generate_and_draw_Minisat22_only_draw_median.py
   ```


## 3.2 `invoke_anthropic`, `invoke_deepseek`, `invoke_openai`
These directories contain the core LLM evaluation pipelines supporting **Figure 1(b)**, **Figure 2**, and related appendix materials.

* **Functionality**:

  * constructs prompts,
  * sends inference requests,
  * parses model outputs,
  * stores prediction results,
  * computes metrics.
* **Metrics**:

  * Accuracy
  * Precision / Recall / F1
  * MCC
  * **ADR**


### Figure 1(b): Correct rate of low-𝛼 UNSAT CNFs’ prediction across models as 𝑁 increases.

The pipelines for reproducing **Figure 1(b)** are organized by model provider.

#### OpenAI models

1. **Input directories**  
   Input CNF instances are stored in directories matching:

    ```regex
   ^Code/invoke_openai/unsat_figure_1b/unsat_cnf_low_alpha_N_[0-9]{1,2}$
    ```

2. **LLM invocation script**
   The script used to query OpenAI models is:

   ```text
   Code/invoke_openai/unsat_figure_1b/invoke_openai_predict_low_alpha_unsat_openai_1.0.0_multiple_model.py
   ```

3. **Prediction output directories**
   Model predictions are written to directories matching:

   ```regex
   ^Code/invoke_openai/unsat_figure_1b/unsat_cnf_low_alpha_N_[0-9]{1,2}_openai_prediction_{LLM_name}$
   ```

4. **Accuracy / correct-rate computation**
   The script used to compute the correct rates needed for **Figure 1(b)** is:

   ```text
   Code/invoke_openai/unsat_figure_1b/calculate_accuracy.py
   ```

---

#### DeepSeek models

1. **Input directories**
   Input CNF instances are stored in directories matching:

   ```regex
   ^Code/invoke_deepseek/unsat_cnf_low_alpha_N_[0-9]{1,2}$
   ```

2. **LLM invocation script**
   The script used to query DeepSeek models is:

   ```text
   Code/invoke_deepseek/invoke_deepseek_predict_low_alpha_unsat.py
   ```

3. **Prediction output directories**
   Model predictions are written to directories matching:

   ```regex
   ^Code/invoke_deepseek/unsat_cnf_low_alpha_N_[0-9]{1,2}_deepseek_prediction_{LLM_name}$
   ```

4. **Accuracy / correct-rate computation**
   The script used to compute the correct rates needed for **Figure 1(b)** is:

   ```text
   Code/invoke_deepseek/calculate_accuracy.py
   ```

---

#### Anthropic models

1. **Input directories**
   Input CNF instances are stored in directories matching:

   ```regex
   ^Code/invoke_anthropic/unsat_cnf_low_alpha_N_[0-9]{1,2}$
   ```

2. **LLM invocation script**
   The script used to query Anthropic models is:

   ```text
   Code/invoke_anthropic/invoke_anthropic_predict_low_alpha_unsat.py
   ```

3. **Prediction output directories**
   Model predictions are written to directories matching:

   ```regex
   ^Code/invoke_anthropic/unsat_cnf_low_alpha_N_[0-9]{1,2}_anthropic_prediction_{LLM_name}$
   ```

4. **Accuracy / correct-rate computation**
   The script used to compute the correct rates needed for **Figure 1(b)** is:

   ```text
   Code/invoke_anthropic/calculate_accuracy.py
   ```


#### Combine All LLMs result and draw Figure 1(b)
```text
"Code/SAT_Draw_Figure/draw_curve_unsat.py"
```


### Figure 2: Comparison of LLM performance across ten metrics

The pipeline for reproducing **Figure 2** is organized as follows.

1. **LLM input data**  
   The CNF inputs used in this experiment are stored in:

   ```text
   Code/invoke_openai/cnf_results_openai_o1
    ```
2. **LLM invocation scripts**
   The scripts used to query different model families are:

   * **OpenAI models**

     ```text
     Code/invoke_openai/invoke_openai_draw_3_curve_multi_models.py
     ```

   * **DeepSeek models**

     ```text
     Code/invoke_openai/invoke_deepseek_predict.py
     ```

   * **Anthropic models**

     ```text
     Code/invoke_openai/invoke_anthropic_predict.py
     ```

3. **Prediction outputs**
   The prediction results produced by the LLMs are written to directories of the form:

   ```text
   Code/invoke_openai/cnf_results_openai_o1_input_based_{LLM_name}
   ```

4. **Second-part data collection**
   To collect the results for the second part of the experiment, the corresponding scripts with the same filenames should also be run under:

   ```text
   invoke_openai/draw_o1_phase_transition_figure
   ```

5. **Metric computation and intermediate summaries**
   The scripts used to compute traditional metrics, ADR, and intermediate summary files are:

   ```text
   Code/invoke_openai/invoke_openai_draw_3_curve_step4_merger_different_models_3_Multiply_3.py
   ```

   and

   ```text
   Code/invoke_openai/merge_two_kinds_of_dataset.py
   ```

6. **Plotting Figure 2**
   The final script used to generate **Figure 2** from the intermediate files is:

   ```text
   invoke_openai/merge_two_kinds_of_dataset_plot_from_intermediate_file.py
   ```





## 3.3 `fix_cnf`

This directory implements the **3-SAT paired-evaluation** workflow used for **Figure 3**.

* **Functionality**:

  * generates hard 3-SAT CNFs,
  * constructs minimally modified SAT/UNSAT counterparts,
  * stores paired inputs,
  * evaluates model discrimination using **ADR**.




### Figure 3: Paired 3-SAT evaluation with ADR

The pipeline for reproducing **Figure 3** is organized as follows.

1. **Generate paired 3-SAT CNF inputs**  
   The following script is used to generate paired CNF instances for the 3-SAT experiments:

   ```text
   Code/fix_cnf/fix_cnf_method_2_remove_mul_minimum_clause.py
    ```

2. **Location of the generated pair inputs**
   The generated paired inputs are stored in:

   ```text
   Code/fix_cnf/pairs_input
   ```

   These files correspond to the same paired inputs used in directories of the form:

   ```text
   Code/fix_cnf/fixed_set_mul_N/unsat_cnf_low_alpha_N_{N}_openai_prediction_o1
   ```

3. **Invoke LLMs to obtain predictions**
   The scripts used to query LLMs and collect prediction results are:

   * **DeepSeek models**

     ```text
     Code/fix_cnf/invoke_deepseek_predict_fixed_set_mul_N.py
     ```

   * **OpenAI models**

     ```text
     Code/fix_cnf/invoke_openai_predict_fixed_set_mul_N.py
     ```

4. **Location of prediction results**
   The prediction outputs are stored in directories of the following forms:

   * **OpenAI prediction results**

     ```text
     Code/fix_cnf/fixed_set_mul_N/unsat_cnf_low_alpha_N_{N}_openai_prediction_o1_openai_prediction_{model_selected}
     ```

   * **DeepSeek prediction results**

     ```text
     Code/fix_cnf/fixed_set_mul_N/unsat_cnf_low_alpha_N_{N}_openai_prediction_o1_deepseek_prediction_{model_selected}
     ```

5. **Metric computation**
   The following script is used to compute traditional evaluation metrics together with **ADR**:

   ```text
   Code/fix_cnf/calculate_3_ways_evaluation.py
   ```

6. **Plotting Figure 3**
   The following script is used to generate **Figure 3**:

   ```text
   Code/SAT_Draw_Figure/draw_curve_pairs.py
   ```







## 3.4 `CNF2`

This directory implements the **2-SAT paired-evaluation** workflow used for **Figure 4**.

* **Functionality**:

  * generates 2-SAT instances,
  * creates paired SAT/UNSAT variants,
  * evaluates traditional metrics and ADR,
  * compares 2-SAT performance against 3-SAT.



### Figure 4: Paired 2-SAT evaluation

The pipeline for reproducing **Figure 4** is organized as follows.

1. **Generate UNSAT 2-SAT CNFs**  
   The following script is used to generate UNSAT 2-SAT instances:

   ```text
   CNF2/generate/CNF2_generate_and_draw_Minisat22.py
    ```

2. **Minimally repair each UNSAT 2-SAT formula**
   After generating the UNSAT 2-SAT instances, a minimal-repair step is applied to construct the corresponding SAT counterparts, thereby forming paired 2-SAT instances.
     ```text
   CNF2/generate/fix_cnf_method_2_remove_mul_minimum_clause.py
    ```    

3. **Storage location for generated and repaired instances**
   The results of the generation and minimal-repair steps are stored in:

   ```text
   Code/CNF2/generate/cnf_results_CDCL
   ```

4. **Invoke OpenAI, DeepSeek, and Anthropic models**
   The scripts used to query the three model families are:

   * **Anthropic models**

     ```text
     C:\Research\Vulnerability\FSE\Satisfiability_Solvers\Code\CNF2\invoke\invoke_anthropic_predict_2SAT.py
     ```

   * **DeepSeek models**

     ```text
     CNF2/invoke/invoke_deepseek_predict_2SAT.py
     ```

   * **OpenAI models**

     ```text
     CNF2/invoke/invoke_openai_predict_2SAT.py
     ```

5. **Prediction outputs**
   The prediction results produced by the LLMs are stored in:

   ```text
   CNF2/generate/cnf_results_CDCL/prediction_result
   ```

6. **Metric computation**
   The scripts used to compute the traditional metrics and **ADR** are:

   ```text
   CNF2/invoke/calculate_3_ways_evaluation.py
   ```

   and

   ```text
   CNF2/invoke/calculate_accuracy.py
   ```

7. **Plotting Figure 4**
   The script used to generate **Figure 4**, which shows the values and trends of multiple metrics on paired 2-SAT instances, is:

   ```text
   SAT_Draw_Figure/draw_curve_2sat.py
   ```




## 3.5 `convert_cnf_to_vertex_cover`

This directory implements the CNF-to-Vertex-Cover reduction used in the cross-representation experiments.

* **Functionality**:

  * converts CNF formulas into graph instances,
  * invokes LLMs on the resulting Vertex Cover tasks,
  * validates returned covers with a deterministic checker,
  * compares decisions and correctness across representations.



### Appendix Figure 1: Vertex Cover results across models. 

The pipeline for reproducing **Appendix Figure 1** is organized as follows.

1. **Input paired CNF instances**  
   The paired CNF inputs used in this experiment are stored in:

   ```text
   Code/fix_cnf/pairs_input
    ```

2. **Convert paired CNFs to Vertex Cover instances**
   The following script is used to convert the paired CNF instances into Vertex Cover instances:

   ```text
   convert_cnf_to_vertex_cover/convert_cnf_to_vertex_cover_method_2.py
   ```

3. **Location of converted Vertex Cover instances**
   The converted results are stored in:

   ```text
   Code/convert_cnf_to_vertex_cover/vertex_cover_graph
   ```

4. **Invoke LLMs on the converted Vertex Cover instances**
   The following scripts are used to query LLMs on the Vertex Cover representations and collect their predictions on whether the original instance is SAT or UNSAT:

   * **OpenAI models**

     ```text
     convert_cnf_to_vertex_cover/convert_cnf_to_vertex_cover_method_1.py
     ```

   * **DeepSeek models**

     ```text
     convert_cnf_to_vertex_cover/convert_cnf_to_vertex_cover_method_1_invoke_deepseek.py
     ```

   The corresponding prediction results are stored in:

   ```text
   Code/convert_cnf_to_vertex_cover/vertex_cover_graph
   ```

5. **Metric computation**
   The following script is used to compute traditional evaluation metrics together with **ADR**:

   ```text
   convert_cnf_to_vertex_cover/calculate_3_ways_evaluation.py
   ```

6. **Plotting Appendix Figure 1**
   The following script is used to generate **Appendix Figure 1**:

   ```text
   SAT_Draw_Figure/draw_curve_vertex_cover.py
   ```




## 3.6 `convert_cnf_to_3D_packing`

This directory implements the CNF-to-discrete-3D-packing reduction used in the cross-representation experiments.

* **Functionality**:

  * converts CNF formulas into discrete packing instances,
  * invokes LLMs using structured prompts,
  * validates returned packing witnesses,
  * compares decision consistency and correctness across representations.


### Appendix Figure 2: 3D packing: predictions when LLM answers “yes” (3×3).

The pipeline for reproducing **Appendix Figure 2** is organized as follows.

1. **Input paired CNF instances**  
   The paired CNF inputs used in this experiment are stored in:

   ```text
   Code/fix_cnf/pairs_input
    ```

2. **Convert paired CNFs to discrete 3D packing instances**
   The following script is used to convert the paired CNF instances into discrete 3D packing instances:

   ```text
   Code/convert_cnf_to_3D_packing/convert_cnf_to_3D_packing_method.py
   ```

3. **Location of converted 3D packing instances**
   The converted results are stored in:

   ```text
   Code/convert_cnf_to_3D_packing/cnf_to_3D_packing
   ```

4. **Invoke LLMs on the converted 3D packing instances**
   The following scripts are used to query LLMs on the 3D packing representations and collect their predictions on whether the original instance is SAT or UNSAT:

   * **OpenAI models**

     ```text
     Code/convert_cnf_to_3D_packing/convert_cnf_to_3D_packing_method_gpt_5.py
     ```

   * **DeepSeek models**

     ```text
     Code/convert_cnf_to_3D_packing/convert_cnf_to_3D_packing_method_deepseek.py
     ```

   * **DeepSeek-Reasoner models**

     ```text
     Code/convert_cnf_to_3D_packing/convert_cnf_to_3D_packing_method_deepseek_reasoner.py
     ```

   The corresponding prediction results are stored in:

   ```text
   Code/convert_cnf_to_3D_packing/cnf_to_3D_packing
   ```

5. **Metric computation**
   The following script is used to compute traditional evaluation metrics together with **ADR**:

   ```text
   Code/convert_cnf_to_3D_packing/calculate_3_ways_evaluation.py
   ```

6. **Plotting Appendix Figure 2**
   The following script is used to generate **Appendix Figure 2**:

   ```text
   Code/SAT_Draw_Figure/draw_3d_packing.py
   ```






## 3.7 `SAT_Draw_Figure`

This directory contains the plotting utilities used to generate publication-quality figures.

* **Functionality**:

  * loads aggregated intermediate results,
  * renders the figures used in the paper and appendix,
  * standardizes fonts, legends, axis ranges, and visual formatting.

---

# 4. Datasets and Instance Generation

The paper includes multiple experimental settings, each with its own instance-construction protocol. This repository supports both direct reuse of prepared data and regeneration from scripts.

## 4.1 Random 3-SAT phase-transition setting

This setting corresponds to the initial experiment on traditional metrics.

* Fixed variable count:

  * (N = 75)
* Clause-density values:

  * (\alpha \in {3.5, 4.0, 4.5, 5.0, 5.5})
* For each (\alpha), the experiment uses multiple randomly generated 3-SAT instances verified by a classical SAT solver.

## 4.2 Low-(\alpha) UNSAT stress-test setting

This setting probes whether LLMs can detect UNSAT instances when satisfiable instances usually dominate.

* Variable counts:

  * (N \in {5, 8, 10, 25, 50, 60, 70, 80, 90, 100, 110, 120, 140})
* Clause-density values:

  * (\alpha \in {3.5, 3.6, 3.7, 3.8, 3.9, 4.0})
* All instances in this setting are solver-verified **UNSAT** formulas.

## 4.3 Paired 3-SAT setting

For each UNSAT formula, the artifact constructs a near-identical SAT counterpart using a minimal edit, such as:

* deleting one clause,
* flipping the polarity of one literal,
* replacing one literal with another.

These pairs are used to evaluate **ADR**, which requires simultaneous correctness on both members of a pair.

## 4.4 Paired 2-SAT setting

The repository also provides generation and evaluation pipelines for paired 2-SAT instances, enabling comparison between a polynomial-time SAT fragment and the harder 3-SAT case.

## 4.5 Cross-representation datasets

The repository includes reduced instances for:

* **Vertex Cover**
* **discrete 3D packing**

These are derived from SAT instances and are used to study **representation-invariant reasoning**.

## 4.6 Prepared vs. regenerated data

If you want to rerun generation yourself, please use the corresponding scripts inside:

* `invoke_traditional_methond`
* `fix_cnf`
* `CNF2`
* `convert_cnf_to_vertex_cover`
* `convert_cnf_to_3D_packing`

---

# 5. LLMs API Configuration


## 5.1 API keys

To rerun commercial-model inference, set the required API keys in your shell environment before execution.

Example:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Depending on your local scripts, you may also choose to place these values in a `.env` file or in provider-specific configuration files.

## 5.2 Important note on cost

The LLMs evaluated in this paper are **closed-source commercial models**. 
Reproducing all model calls therefore requires **paid API access**. 
For this reason, the default artifact workflow focuses on reproducing results from 
prepared intermediate files rather than rerunning all LLM calls.

---

# 6. Environment Requirements

## 6.1 Recommended environment

* Python 3.10 or newer
* Linux, macOS, or Windows

## 6.2 Required Python packages

The artifact depends on two classes of packages: core packages and API packages.

### 6.2.1 Core dependencies

These are used for data generation, metric computation, validation, and plotting:

* `pandas`
* `numpy`
* `matplotlib`
* `openpyxl`
* `python-sat`
* `tqdm`

### 6.2.2 API dependencies

These are used for invoking commercial LLM services:

* `openai`
* `anthropic`
* `requests`

Install all required packages with:

```bash
pip install scikit-learn, pandas numpy matplotlib openpyxl openai anthropic requests python-sat tqdm
```

---

# 7. Reproducing Main-Paper Results

Because the evaluated LLMs are commercial models, the default workflow focuses on reproducing results from the provided intermediate files and processed outputs.

## 7.1 Reproduce all figures

```bash
python reproduce_all_figures.py
```

## 7.2 Figure 1(a): CDCL phase-transition baseline

* **Plotting code**:
  `Code/invoke_traditional_methond/phase_transition/phase_transition_generate_and_draw_Minisat22_only_draw_median.py`
* **Prediction / solver results**:
  `Code/invoke_traditional_methond/phase_transition/cnf_results_CDCL`
* **Saved Figure path**:
   `Analysis_Result_Collection/Figure_in_paper/phase_transition/figures_CDCL_phase_transition/Random_3-SAT_CDCL_N_75_median.png`

## 7.3 Figure 1(b): low-(\alpha) UNSAT detection across (N)

* **Aggregated intermediate data**:
  `Code/Analysis_Result_Collection/SAT_figures.xlsx`
* **Plotting code**:
  `Code/SAT_Draw_Figure/draw_curve_unsat.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/unsat/unsat_small_alpha_prediction_correct_rate.png`


## 7.4 Figure 2: traditional metrics across clause densities

* **Aggregated intermediate data**:
  `Analysis_Result_Collection/Figure_in_paper/LLMs_phase_transition/combined_metrics/combined_metrics_by_model_alpha.xlsx`
* **Plotting code**:
  `Code/invoke_openai/merge_two_kinds_of_dataset_plot_from_intermediate_file.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/LLMs_phase_transition/combined_metrics/combined_metrics_4x3.png`

## 7.5 Figure 3: paired 3-SAT evaluation with ADR

* **Aggregated intermediate data**:
  `Code/Analysis_Result_Collection/SAT_figures.xlsx`
* **Plotting code**:
  `Code/SAT_Draw_Figure/draw_curve_pairs.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/pairs/pairs_small_alpha_prediction_correct_rate_3x3_only.png`


## 7.6 Figure 4: paired 2-SAT evaluation

* **Aggregated intermediate data**:
  `Code/Analysis_Result_Collection/SAT_figures.xlsx`
* **Plotting code**:
  `Code/SAT_Draw_Figure/draw_curve_2sat.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/2SAT/2SAT_3x3_only.png`

## 7.7 Appendix Figure 1: Vertex Cover results

* **Aggregated intermediate data**:
  `Code/Analysis_Result_Collection/SAT_figures.xlsx`
* **Plotting code**:
  `Code/SAT_Draw_Figure/draw_curve_vertex_cover.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/Vertex_Cover/Vertex_Cover_legend_in_last_panel.png`


## 7.8 Appendix Figure 2: 3D packing results

* **Aggregated intermediate data**:
  `Code/Analysis_Result_Collection/SAT_figures.xlsx`
* **Plotting code**:
  `Code/SAT_Draw_Figure/draw_3d_packing.py`
* **Saved Figure path**:
  `Analysis_Result_Collection/Figure_in_paper/3d_packing/packing_pred_llm_yes_3x3.png`

---


# 12. Summary Table

| Directory                     | Scope                                       | Paper Reference     |
| :---------------------------- | :------------------------------------------ |:--------------------|
| `invoke_traditional_methond`  | CDCL solver baselines                       | Figure 1(a)         |
| `invoke_{llm_provider}`       | LLM inference, parsing, metrics, ADR        | Figures 1(b), 2     |
| `fix_cnf`                     | 3-SAT pair generation and paired evaluation | Figure 3            |
| `CNF2`                        | 2-SAT pair generation and paired evaluation | Figure 4            |
| `convert_cnf_to_vertex_cover` | CNF to Vertex Cover reduction               | Section 7, Appendix |
| `convert_cnf_to_3D_packing`   | CNF to discrete 3D packing reduction        | Section 8, Appendix |
| `SAT_Draw_Figure`             | Visualization scripts                       | All figures         |

---


