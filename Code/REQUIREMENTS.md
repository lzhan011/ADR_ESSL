# REQUIREMENTS

## Artifact title
**Evaluating Satisfiability Solving with LLMs**

## Purpose of this artifact
This artifact supports reproducibility for the paper *Evaluating Satisfiability Solving with LLMs*. It contains scripts and prepared intermediate results for regenerating the main figures and appendix figures, as well as partial end-to-end pipelines for instance generation, commercial-LLM invocation, witness validation, and metric computation.

The recommended evaluation path is to reproduce the figures directly from the provided intermediate files, without rerunning all commercial-model inference.

---

## 1. Hardware requirements

### Recommended minimum hardware
The default reproduction path is lightweight and does **not** require GPUs.

A typical commodity machine should be sufficient, for example:

- 64-bit CPU
- 8 GB RAM or more recommended
- at least 5 GB of available disk space recommended for the artifact files, generated figures, and intermediate outputs

### Notes
- No non-standard peripherals are required.
- No cluster, VM, or GPU is required for the default quick reproduction path.
- Re-running all commercial-model inference may take substantially longer and depends on external API services rather than local hardware alone.

---

## 2. Software requirements

### Operating system
The artifact is intended to run on:

- Linux
- macOS
- Windows

Linux or macOS is recommended for the smoothest command-line workflow.

### Python
- Python **3.10 or newer**

### Python packages
The artifact depends on the following Python packages:

#### Core dependencies
These are required for the recommended reproduction path:

- `pandas`
- `numpy`
- `matplotlib`
- `openpyxl`
- `python-sat`
- `tqdm`
- `scikit-learn`

#### API-related dependencies
These are required only if the user wants to rerun commercial-model inference:

- `openai`
- `anthropic`
- `requests`

### Installation command
A minimal installation can be done with:

```bash
pip install scikit-learn, pandas numpy matplotlib openpyxl openai anthropic requests python-sat tqdm
````

---

## 3. External services and network requirements

### Default artifact evaluation path

The recommended quick reproduction path does **not** require external API calls, provided that the included intermediate files are used.

### Optional commercial-model reruns

To rerun the commercial-model inference pipelines, the following are required:

* Internet access
* valid API keys for the corresponding providers
* paid access to the relevant services, if applicable

The following environment variables may be needed:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Because the evaluated LLMs are closed-source commercial models, full end-to-end reruns of all model calls may incur monetary cost and may also depend on provider-side model/version availability.

---

## 4. Included data and supported reproduction modes

This artifact supports two main modes of use.

### 4.1 Recommended mode: reproduction from prepared intermediate files

Most main-paper and appendix figures can be regenerated directly from prepared `.xlsx` files and included processed outputs.

This is the recommended mode for artifact evaluation.

### 4.2 End-to-end reruns

The artifact also includes scripts for:

* generating SAT / UNSAT instances
* generating paired SAT / UNSAT instances
* invoking commercial LLMs
* parsing predictions
* validating witnesses
* recomputing traditional metrics and ADR

These workflows are provided for transparency and traceability, but they are not required for the basic artifact check.

---

## 5. Expected runtime

### Quick reproduction path

Running the default top-level reproduction script is expected to be relatively lightweight on a standard machine:

```bash
python reproduce_all_figures.py
```

Around 10 minutes, actual runtime depends on the host machine and local Python environment, but the quick path is intended to be practical for artifact evaluation.

### Commercial-model reruns

End-to-end reruns that invoke commercial LLM APIs may take much longer and depend on:

* network conditions
* API rate limits
* service availability
* model response latency
* paid usage limits

---

## 6. Special environment notes

* No Docker image is required for the default reproduction path.
* No virtual machine is required for the default reproduction path.
* The artifact includes plotting scripts, aggregated intermediate spreadsheets, selected prediction-result directories, prompts, and supplementary materials used to reproduce the figures in the paper and appendix.
* The artifact is designed so that the main empirical claims can be checked without rerunning all original LLM calls.

---

## 7. Main output targets

The quick reproduction path is intended to regenerate figures such as:

* Figure 1(a): CDCL phase transition
* Figure 1(b): low-α UNSAT detection
* Figure 2: comparison across metrics
* Figure 3: paired 3-SAT evaluation with ADR
* Figure 4: paired 2-SAT evaluation
* Appendix Figure 1: Vertex Cover results
* Appendix Figure 2: 3D packing results

Representative output locations include:

* `Analysis_Result_Collection/Figure_in_paper/phase_transition/figures_CDCL_phase_transition/Random_3-SAT_CDCL_N_75_median.png`
* `Analysis_Result_Collection/Figure_in_paper/unsat/unsat_small_alpha_prediction_correct_rate.png`
* `Analysis_Result_Collection/Figure_in_paper/LLMs_phase_transition/combined_metrics/combined_metrics_4x3.png`
* `Analysis_Result_Collection/Figure_in_paper/pairs/pairs_small_alpha_prediction_correct_rate_3x3_only.png`
* `Analysis_Result_Collection/Figure_in_paper/2SAT/2SAT_3x3_only.png`
* `Analysis_Result_Collection/Figure_in_paper/Vertex_Cover/Vertex_Cover_legend_in_last_panel.png`
* `Analysis_Result_Collection/Figure_in_paper/3d_packing/packing_pred_llm_yes_3x3.png`

---

## 8. Summary

In summary, this artifact requires only a standard Python environment for the default reproduction workflow. No GPU and no special hardware are needed for the recommended path. Commercial-model reruns are optional and require additional API credentials, network connectivity, and potentially paid access.


