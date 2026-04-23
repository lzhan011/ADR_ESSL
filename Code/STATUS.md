
# STATUS

## Artifact title
**Evaluating Satisfiability Solving with LLMs**

## Requested badges

The authors apply for the following badges:

1. **Available**
2. **Evaluated – Functional**

---

## 1. Badge requested: Available

We believe this artifact deserves the **Available** badge for the following reasons:

- The artifact is publicly released in a permanent archival repository.
- The artifact is distributed under an open-source license.
- The artifact package includes the materials needed for evaluation and reproduction, including source code, prepared intermediate files, selected prediction-result directories, plotting scripts, and supplementary materials.
- The artifact is accompanied by documentation files, including `README`, `REQUIREMENTS`, `INSTALL`, and this `STATUS` file.

In particular, the artifact contains:

- CNF generation,
- paired-CNF generation,
- LLM API invocation pipelines,
- traditional evaluation metrics,
- the proposed ADR metric,
- cross-representation reductions to Vertex Cover and discrete 3D packing,
- deterministic witness validation,
- plotting scripts and prepared intermediate results needed to reproduce the main findings of the paper.

These materials together make the artifact publicly accessible and usable as a reproducibility package for the paper.

---

## 2. Badge requested: Evaluated – Functional

We believe this artifact deserves the **Evaluated – Functional** badge for the following reasons:

### 2.1 The artifact is documented

The artifact is documented through the provided `README`, `REQUIREMENTS`, and `INSTALL` files.

In particular, the `README` documents the artifact in substantial detail. It explains:

- the scope of the artifact,
- the recommended quick-start workflow,
- the repository structure,
- the purpose of the major directories,
- the main functionality of the core scripts,
- the step-by-step workflows used to reproduce the figures in the paper and appendix,
- the required dependencies,
- the optional API-key configuration for commercial-model reruns, and
- the expected output locations of the generated figures.

This detailed documentation makes the artifact easier to understand, navigate, and exercise during evaluation.

### 2.2 The artifact is complete enough to exercise the main functionality

The artifact contains the main components needed to exercise the core workflows used in the paper, including:

- generation of SAT / UNSAT instances,
- generation of paired SAT / UNSAT instances,
- commercial-model invocation scripts,
- prediction parsing,
- traditional metric computation,
- ADR computation,
- conversion to Vertex Cover and discrete 3D packing representations,
- deterministic witness validation, and
- plotting utilities for the figures in the paper and appendix.

The README also explains that the artifact supports both:

- fast reproduction from aggregated intermediate files, and
- partial end-to-end reruns from prediction-result directories and included scripts.

### 2.3 The artifact provides a clear and practical evaluation path

The README provides an explicit top-level quick-start command:

```bash
python reproduce_all_figures.py
````

which is described as the fastest way to verify the main empirical claims of the paper.

Moreover, the recommended evaluation path does not require rerunning all commercial-model inference. Instead, the README explains that most main-paper and appendix figures can be regenerated directly from the provided intermediate `.xlsx` files and processed outputs. This makes the artifact directly exercisable for evaluation.

### 2.4 The artifact explains file structure, functionality, and workflow steps in detail

A key reason we believe the artifact deserves the **Evaluated – Functional** badge is that the `README` does not merely list files. It also explains:

* how the repository is organized by experimental setting,
* what each major directory is responsible for,
* what the major scripts do,
* how the scripts relate to the figures in the paper,
* where the inputs are located,
* where prediction results are stored,
* which scripts compute metrics, and
* which scripts generate the final plots.

For each major experiment, the README provides a concrete workflow that connects inputs, processing steps, metric computation, and plotting. This level of explanation substantially improves the usability of the artifact for evaluators.

### 2.5 The artifact identifies concrete scripts and expected outputs

The README maps the major results to concrete scripts and output files. In particular, it specifies scripts and saved output paths for:

* Figure 1(a),
* Figure 1(b),
* Figure 2,
* Figure 3,
* Figure 4,
* Appendix Figure 1, and
* Appendix Figure 2.

This allows evaluators to run either the full quick-reproduction script or selected figure-generation scripts and then check whether the documented output files are produced in the expected locations.

### 2.6 The artifact includes environment and dependency guidance

The documentation specifies:

* the intended operating systems,
* the recommended Python version,
* the required Python packages,
* the installation command, and
* the optional environment variables needed for commercial-model reruns.

This supports direct installation and execution by artifact evaluators.

---

## 3. Intended evaluation mode

The recommended evaluation workflow for this artifact is:

1. install the required Python dependencies,

2. follow the setup instructions in `INSTALL`,

3. run

   ```bash
   python reproduce_all_figures.py
   ```

4. verify that the expected figure files are regenerated in the documented output directories.

This evaluation path exercises the main artifact functionality in a practical way and reproduces the principal empirical results of the paper from the included intermediate files.

---

## 4. Scope clarification

This artifact is primarily designed to support **reproducibility of the paper’s results**.

The default evaluation path:

* does not require GPUs,
* does not require special hardware,
* does not require rerunning all commercial-model inference, and
* does not require access to external paid APIs.

Commercial-model reruns are included as optional workflows for transparency and traceability, but they are not necessary for the main artifact evaluation path.

---

## 5. Summary

In summary, the authors request:

* **Available**, because the artifact is publicly archived, openly distributed, and accompanied by the materials and documentation needed for evaluation and reuse; and
* **Evaluated – Functional**, because the artifact is documented, complete enough to exercise the core workflows of the paper, and provides a clear, practical, and executable path for reproducing the main figures and empirical findings.


