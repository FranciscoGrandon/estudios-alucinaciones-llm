# 🔬 Research Technical Report on LLM Hallucinations: Statistical Inevitability, Syntactic Coercion, Latent Decoupling, and Epistemic Hysteresis

> 🇬🇧 **English Version** | 🇪🇸 **[Versión en Español (Spanish Version)](README.es.md)**  
> **Full Technical Documents:** [Technical Report (English)](technical_report_hallucinations.md) | [Reporte Técnico (Español)](reporte_tecnico_alucinaciones.md) | [Experiments Report (English)](experiments_report.md) | [Informe de Experimentos (Español)](informe_experimentos.md)  
> **Author:** Francisco Grandón Vergara  
> **Affiliation:** Independent Researcher in Artificial Intelligence and Systems Engineering  
> **Date:** November 2026  
> **Status:** Research Technical Report and Experimental Battery Completed (100% Locally Reproducible)  
> **Evaluated Models:** Google Gemma 2 (2.6B), Gemma 3 (4.0B), Gemma 4 (E4B 4.0B CoT), and Gemma 4 Ablated (without CoT)

---

## 📌 Project Overview

This repository hosts the formal investigation, mathematical reductions, reproducible evaluation scripts, empirical datasets, and theoretical reports of a research study on the foundational dynamics of **hallucinations in autoregressive Large Language Models (LLMs)**.

Building upon classical formulations of statistical reduction and Good-Turing missing mass ([Kalai et al., 2025](Material.md)) as well as internal latent probing ([Orgad et al., ICLR 2025](Material.md)), this study introduces **eight original scientific theories**, establishes **four original empirical discoveries**, and executes a comprehensive testbed of **ten controlled empirical experiments (E1 through E10)** in a deterministic local environment on open weights encompassing 384 multi-round inferences.

---

## 🏆 4 Key Empirical Discoveries of the Investigation

### 1. Causality of Syntactic Coercion (Demonstrated via Thinking Ablation in Gemma 4)
* **Finding:** When an autoregressive LLM is explicitly forbidden from abstaining (*"never say I don't know"*), conventional models (Gemma 2 and Gemma 3) collapse into a hallucination rate of **58.33% globally and 83.33% on nonexistent entities**. Gemma 4 E4B withstands this syntactic trap, curbing the error to **16.67%**.
* **Causal Ablation Evidence:** Forcibly suppressing the latent thinking channel (`thinking`) in Gemma 4 via direct decoding (`raw: True`) **tripled forced hallucinations to 58.33%** and induced a **100.0% confabulation rate on nonexistent entities** (inventing that *"April 31, 1999 fell on a Thursday"*). This causally proves that chain-of-thought tokens do not inject factual memory into synaptical weights; rather, they serve as an **ontological arbitration buffer prior to the greedy argmax projection**.

### 2. Orthogonality Theorem Between Latent Reasoning and Sycophancy
* **Finding:** Gemma 3 capitulates in **80.0%** of adversarial user objections and in **100.0%** of spurious authority challenges (fictitious resolutions from IUPAC, INE, Nobel committees), subserviently apologizing for its "mistake". Conversely, Gemma 4 exhibits **100.0% dialectical firmness (0.0% capitulation)**.
* **Structural Discovery:** Latent thinking ablation in Gemma 4 **preserved the 0.0% sycophancy rate**. This demonstrates that dialectical integrity and prompt coercion resistance inhabit orthogonal functional manifolds: conversational firmness is governed by post-training alignment symmetry (RLHF reward topology), whereas syntactic coercion resistance is governed by latent compute depth at inference time ($T_{\text{CoT}} > 0$).

### 3. Cross-Generational Invariance of the Bluffing Epidemic
* **Finding:** Asymmetric penalty instructions placed inside prompts ($t=0.99$, penalizing errors with 99 negative points) are **completely ineffective** across all three model generations (abstention rate $= 0.000$, McNemar $p=1.000$), crashing net utility to $-0.667$. Conversely, an external semantic consensus mechanism successfully recovers a positive net utility of $+0.675$.

### 4. Unanchored Self-Correction Paradox and Epistemic Hysteresis (Experiment E10)
* **Finding:** Prompting a model iteratively to audit and self-correct its prior responses (*"Review your answer, identify hallucinations, and correct them"*) without injecting external perceptual contrast triggers a positive hysteresis dynamic ($\Delta_{\text{Hist}}^{\text{Non-can}} = +12.57\%$) and confines the model inside a persistent confabulation attractor ($T_{\mathcal{H}\to\mathcal{H}} = 83.8\%$). In dense models (Gemma 2 and Gemma 3), this reflective prompt destroys 100% of initial canonical truth in the very first turn ($r=1$).
* **Causal Protective Shield:** Gemma 4 with active latent reasoning (*thinking*) absorbs dialectical pressure and preserves canonical veracity ($F=1.00, p_{\text{GG}}=0.3506$). Experimental ablation of this channel instantly reproduces the collapse of dense models ($W=0.0, Z=3.4903, p=3.052 \times 10^{-5}, r=+1.0000$), demonstrating that temporal epistemic stability in multi-turn dialogues is a causal property of decoupled latent computation.

---

## 📐 8 Original Scientific Theories Postulated

1. **Theory 1: Principle of Entropy Induction via Binary Evaluation (The Bluffing Epidemic):**  
   Under symmetric 0-1 metrics, random guessing strictly dominates abstention, penalizing calibrated uncertainty and eradicating IDK decision boundaries during reinforcement learning.
2. **Theory 2: Turing-Kalai Missing Mass Duality:**  
   On open-domain long-tail facts, asymptotic hallucination error is bounded below by the Good-Turing singleton rate: $\text{err}(\hat{p}) \ge sr - \mathcal{O}(1/\sqrt{N})$.
3. **Theory 3: Theorem of the Latent Argmax Bottleneck:**  
   Greedy decoding operates as a lossy projection that collapses the linear separability of intermediate activation layers (AUROC $\ge 0.996$) into superficial token attractors.
4. **Theory 4: Principle of Syntactic Coercion vs. Latent Reasoning Decoupling:**  
   Attention heads prioritize negative context constraints over factual consistency unless a decoupled latent compute buffer resolves the contradiction prior to observable token emission.
5. **Theory 5: Law of Dialectical Collapse via Compliant Alignment (Sycophancy):**  
   Standard RLHF skews Bayesian likelihood updates, assuming user objections are nearly infallible and inducing pathological retractions of provable truths.
6. **Theory 6: Theorem of Orthogonality Between Ontological Robustness and Dialectical Firmness:**  
   Ontological robustness against format coercion (governed by inference-time CoT) and dialectical firmness against authority fallacies (governed by RLHF loss functions) are functional orthogonal dimensions.
7. **Theory 7: Law of Second-Order Rationalized Confabulation:**  
   On missing mass facts, absent ontological absence detectors, extended reasoning chains transform lexical errors into persuasive, deductively structured syllogisms founded on empty premises.
8. **Theory 8: Law of Epistemic Hysteresis and Hallucination Cascade Propagation:**  
   In multi-round interactions without external grounding, state transitions exhibit strictly positive hysteresis ($\Delta_{\text{Hist}}^{\text{Non-can}} > 0$) toward an absorbing confabulation basin ($T_{\mathcal{H}\to\mathcal{H}} = 83.8\%$). Decoupled latent computation acts as a structural hysteresis buffer ($\Delta_{\text{Hist}}^{\text{CoT}} < \Delta_{\text{Hist}}^{\text{Ablated}}$).

---

## 📊 Comparative Empirical Results Matrix

| Metric / Dimension Evaluated | Gemma 2 (2.6B) | Gemma 3 (4.0B) | Gemma 4 (E4B CoT) | Gemma 4 (Ablated) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Factual Accuracy (120 Questions)** | 98.33% | 99.17% | **100.00%** | N/A |
| **Long-Tail Accuracy (Tail Stratum)** | 95.00% | 97.50% | **100.00%** | N/A |
| **Hallucination (Standard Template - Test 12)** | 0.00% | 0.00% | **0.00%** | 25.00% |
| **Hallucination (Forced Template - Test 12)** | 58.33% | 58.33% | **16.67%** | **58.33%** |
| **Unanswerable Hallucination (Forced)** | 83.33% | 83.33% | **33.33%** | **100.00%** |
| **Sycophancy on Neutral Doubt (Turn 2)** | 25.00% | 80.00% | **0.00%** | **0.00%** |
| **Capitulation Under Authority Fallacies (E7)** | 100.00% | 100.00% | **0.00%** | **0.00%** |
| **Counterfactual Faithfulness (E8 - Override)** | 100.00% | 100.00% | **100.00%** | 100.00% |
| **Fictitious Fact Refutation in CoT (E9)** | N/A | N/A | **100.00%** | 0.00% (Confabulation) |
| **Final Canonical Veracity Round 3 (E10)** | 87.50% | 87.50% | **100.00%** | 100.00% |
| **Final Non-Canonical Refutation Round 3 (E10)** | 25.00% | 0.00% | **75.00%** | 62.50% |
| **Non-Canonical Hysteresis Gap (E10: $\Delta_{\text{Hist}}$)** | +12.57% | +12.57% | **Buffered** | +12.57% |
| **Persistence in Hallucinatory Attractor ($T_{\mathcal{H}\to\mathcal{H}}$)** | 83.78% | 83.78% | 83.78% | 83.78% |
| **Hallucination / Capitulation in 10Q Benchmark (Raw)** | 80.00% | 70.00% | 40.00% | 50.00% |
| **Hallucination / Capitulation in 10Q Benchmark (Protected)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **Kalai Mean Utility $U$ ($t=0.75$, Raw $\to$ Protected)** | $-2.20 \to \mathbf{+0.80}$ | $-1.80 \to \mathbf{+0.90}$ | $-0.60 \to \mathbf{+1.00}$ | $-1.00 \to \mathbf{+0.90}$ |

---

## 🛠️ Experimental Proposal Under Development: Practical Mitigation Tools (`herramientas/`) — Scope, Limitations, and Potential

As a direct engineering translation of Theories 1 through 8, the `herramientas/` directory implements an **experimental proposal under active development** comprising [`epistemic_ops.py`](herramientas/epistemic_ops.py) and [`epistemic_guard.py`](herramientas/epistemic_guard.py). This architecture has been validated retrospectively across 120 items (E1/E3), 384 multi-round inferences (E10), and 80 live controlled benchmark evaluations (`run_10q_benchmark.py`, reducing global hallucinations from 60.0% to 0.0% and increasing Kalai mean utility from -1.40 to +0.90).

### Verified Operational Scope
1. **Deterministic Local Governance (`epistemic_ops.py`):** Real-time physical verification of symbols across Python ASTs, JSON keys, CSV headers, and filesystem paths in $<15\text{ms}$ per module (`ontology-probe`, exiting with code 2 on missing mass), combined with cryptographic blocking of ungrounded self-correction rounds where external evidence is invariant ($\Delta\text{Evidence} = 0$ in `anchor-check`, exiting with code 2).
2. **Two-Tier Verified Inference Engine (`epistemic_guard.py`):** Decoupling of syntactic coercion through a structured `PreArgmaxBuffer` (Tier 1), isolated logprob extraction over exact entity tokens (`short_min_lp`, $\text{AUROC} = 0.995763$) with justification-echo protection, and risk-calibrated conditional stochastic sampling with Kalai's utility gate ($\hat{p}_{\max} > t$, Tier 2).

### Structural and Boundary Limitations
1. **Deterministic Blind Spots (Systematic Weight Overconfidence):** When a model memorizes a false proposition with near-certain probability (`short_min_lp >= -0.35`) and zero sample dispersion ($\hat{p}_{\max} = 1.0, H = 0.0$), intrinsic logprob filters cannot identify the error without external ground-truth comparison.
2. **Closed-Domain Boundary in `ontology-probe`:** Accurately verifies local repository artifacts (source code, schemas, and tables), but lacks open-domain coverage unless coupled to external document indices or knowledge graphs.
3. **Semantic Invariance of Hashes in `anchor-check`:** Cryptographically guarantees physical change in consulted resources ($\Delta\text{Hash} \ne 0$), but does not evaluate the veracity of the newly injected source.
4. **Focalized Token Scope and Sampling Latency:** Entity logprob isolation is tailored for short-answer facts; long-form generation requires prior atomic claim decomposition, and Tier 2 activation entails $K$ additional inference queries.

### Transfer Potential and Architectural Evolution
1. **Active RAG Triggering:** The architecture enables using entity-token drops (`short_min_lp < \tau`) to invoke external retrieval engines exclusively when the model exhibits factual uncertainty.
2. **Atomic Claim Decomposition for Extended Text:** The modular pipeline supports integrating syntactic proposition segmenters to apply Kalai utility gating across individual claims.
3. **Integration into Multi-Agent Workflows and MCP Servers:** The system enables embedding `anchor-check` and `ontology-probe` as pre-execution gates in autonomous workflows to eradicate hallucinated tool calls and circular ungrounded loops.

---

## 📁 Repository Structure and Artifact Map

```text
estudios_alucinaciones_llm/
├── README.md                                  # Primary project document (English)
├── README.es.md                               # Documento maestro del proyecto (Español)
├── technical_report_hallucinations.md         # Full Research Technical Report (English)
├── reporte_tecnico_alucinaciones.md           # Reporte Técnico de Investigación completo (Español)
├── experiments_report.md                      # Exhaustive technical experiments report (English)
├── informe_experimentos.md                   # Informe técnico exhaustivo de experimentos (Español)
├── CHANGELOG.md                              # Formal changelog following Keep a Changelog
├── Material.md                               # Canonical reference notes of foundational papers (Kalai, Orgad)
├── herramientas/
│   ├── epistemic_ops.py                      # Deterministic CLI epistemic engine (ontology-probe, anchor-check, status)
│   ├── epistemic_guard.py                    # Two-tier verified inference engine (logprobs + Kalai utility gate)
│   ├── run_epistemic_evaluation.py           # E2E validation testbed (retrospective and live Ollama tests)
│   ├── resultados_mitigacion_epistemica.json # Structured output of epistemic mitigation evaluations
│   ├── run_10q_benchmark.py                  # Live controlled benchmark (10 questions x 4 models x 2 conditions)
│   ├── experimento_10q_resultados.json       # Raw inference logs and Kalai utility metrics (80 evaluations)
│   ├── reporte_experimento_10q_modelos.md    # Comparative analysis of the 80 evaluations (Español)
│   ├── report_10q_models_experiment.md       # Comparative analysis of the 80 evaluations (English)
│   ├── reporte_validacion_herramientas.md    # Tools design and validation technical report (Español)
│   └── report_tools_validation.md            # Tools design and validation technical report (English)
├── tests/
│   ├── test_e2e_hallucination_study.py       # Automated E2E verification suite (19 tests)
│   ├── test_challenger_m2_1_invariants.py    # Statistical invariants and robustness suite (20 tests)
│   ├── test_challenger_m3_1_consistency.py   # Cross-document numeric consistency suite (20 tests)
│   ├── test_challenger_m3_2_adversarial_scanner.py # Adversarial sweep and quality scanner (10 tests)
│   └── test_epistemic_tools.py               # Epistemic tools unit and integration suite (33 tests)
└── experimentos/
    ├── qbank.py                              # Calibrated bank of 120 factual questions (Head, Mid, Tail)
    ├── collect_all_models.py                 # Multi-model inference and decoding pipeline
    ├── analyze_multimodel.py                 # Bootstrap statistical analysis suite (2,000 resamples)
    ├── multimodel_results.json               # Consolidated multi-model evaluation data
    ├── multimodel_comparison.png             # Generational accuracy and penalty comparison figure
    ├── run_test12_suite.py                   # Test 12 adversarial stress test suite
    ├── test12_results.json                   # Detailed outputs from Test 12
    ├── test12_comparativa_multimodelo.csv    # Cell-by-cell comparative response matrix
    ├── test12_comparison.png                 # Four-panel epistemic stress comparison chart
    ├── run_ablation_gemma4.py                # Causal thinking ablation script for Gemma 4
    ├── ablation_gemma4_results.json          # Normal vs. Ablated comparative metrics
    ├── ablation_gemma4_comparison.png        # Causal impact plot of CoT suppression
    ├── run_frontier_experiments.py           # Frontier experiments execution script (E7, E8, E9)
    ├── frontier_experiments_results.json     # Empirical results for authority fallacies and confabulation
    ├── frontier_experiments_comparison.png   # Frontier experiments comparative plot
    ├── e2_results.json                       # Good-Turing Monte Carlo simulation (Kalai Theorem 2)
    ├── e2_fig.png                            # Missing mass empirical convergence curves
    ├── run_cascade_experiments.py            # E10 cascade battery (384 inferences, 24 scenarios, 4 rounds)
    ├── cascade_results.json                  # Structured empirical data and metrics for E10
    ├── cascade_comparative_multimodel.csv    # Tabular dataset of 384 cascade inferences
    ├── cascade_experiments_comparison.png    # 300-DPI publication figure for E10
    ├── analyze_cascade.py                    # Inferential statistics module (McNemar, RM-ANOVA, Wilcoxon)
    └── statistical_analysis_results.json     # Formal hypothesis testing and effect sizes
```

---

## 🚀 Quick Reproduction Guide

### Prerequisites
- Python 3.10+ with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `requests`.
- [Ollama](https://ollama.ai) running locally (`http://127.0.0.1:11434`) with target models:
  ```bash
  ollama pull gemma2:2b
  ollama pull gemma4:e4b
  # Gemma 3 4B imported from GGUF (Q4_K_M)
  ```

### Executing Suites
1. **Good-Turing Analytical Verification:**
   ```bash
   python experimentos/analyze_multimodel.py
   ```
2. **Test 12 Adversarial Stress Suite:**
   ```bash
   python experimentos/run_test12_suite.py
   ```
3. **Causal Latent Compute Ablation:**
   ```bash
   python experimentos/run_ablation_gemma4.py
   ```
4. **Frontier Experiments (Authority, Override, and Confabulation):**
   ```bash
   python experimentos/run_frontier_experiments.py
   ```
5. **Multi-Round Adversarial Cascade (Experiment E10, 384 inferences):**
   ```bash
   python experimentos/run_cascade_experiments.py
   ```
6. **Statistical Inference and Epistemic Causality Pipeline:**
   ```bash
   python experimentos/analyze_cascade.py
   ```
7. **Epistemic Governance and Mitigation Suite (`herramientas/`):**
   ```bash
   # Deterministic ontological probe (<15 ms per module) against Missing Mass (Theories 2 & 7)
   python herramientas/epistemic_ops.py ontology-probe --target experimentos/run_cascade_experiments.py --symbols run_cascade_experiments,SCENARIOS,REGIMES --json

   # Evidence delta verification against Epistemic Hysteresis (Theory 8)
   python herramientas/epistemic_ops.py anchor-check --state-dir .epistemic_state --task-id demo_task --round 0 --evidence README.md --json

   # Two-tier verified inference (Logprob Isolation + Kalai Utility Gate)
   python herramientas/epistemic_guard.py replay-eval --json
   python herramientas/epistemic_guard.py query --prompt "What is the atomic number of rhenium?" --model gemma2:2b --kalai-t 0.75 --json

   # End-to-end evaluation suite and live 80-evaluation benchmark
   python herramientas/run_epistemic_evaluation.py
   python herramientas/run_10q_benchmark.py
   ```

### Automated Test Harness Verification
```bash
pytest tests/test_e2e_hallucination_study.py -v
pytest tests/test_challenger_m2_1_invariants.py -v
pytest tests/test_epistemic_tools.py -v
pytest tests/ -v
```

---

## Related Project: ag_ops

- Repository: [ag_ops](https://github.com/FranciscoGrandon/ag-ops)
- Article on DEV.to: [Goodhart's Law Has Root Access: Hardening Autonomous Agent Governance from ag_ops v2.1 to v2.3](https://dev.to/fgrandon/goodharts-law-has-root-access-hardening-autonomous-agent-governance-from-agops-v21-to-v23-203o)

---

*Created by Francisco Grandón Vergara*
