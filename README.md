# Can We Trust Logged LLM Comparisons? — reproducibility package

Code, result files and figures for the workshop paper *Can We Trust Logged LLM Comparisons? Auditing
Selection and Measurement Validity under Model Routing*, targeted for the TAE workshop at NeurIPS
2026, *Can We Trust AI Evaluation? Robustness, Causality, and Risk in Modern AI Assessment*
(<https://tai-eval.github.io/>).

This repository is the reproducibility package only, and carries no manuscript source. The paper and
its LaTeX build live in a companion repository, which mirrors everything here. That repository is not
linked while the submission is under double-blind review.

**One line.** A router-generated log can support the wrong model-selection conclusion for two
unrelated reasons, and a randomized audit repairs exactly one of them. This package runs the crossed
experiment that separates the two, on two released routing matrices.

---

## What the experiment does

A production router sends easy requests to cheap models and hard requests to strong ones, then logs
an automatic score for each answer. Reading a model comparison off that log can fail because

* **assignment validity** fails, since the models were scored on different requests, and
* **measurement validity** fails, since the automatic scorer may not measure the outcome the
  deployment decision is about.

The experiment switches each failure on and off independently on the same data, holding the model
roster, requests, contexts and log volume fixed.

|                       | faithful evaluator (D0) | distorted evaluator (D1) |
|-----------------------|-------------------------|--------------------------|
| **common-population log (A0)** | no systematic threat | measurement only |
| **router-selected log (A1)**   | assignment only      | both |

`A1–D2` adds an evaluator that behaves differently in the log and in the audit, as a diagnostic.

---

## Layout

```
code/
  eval_validity/           the experiment package
    grid.py                cohorts, contexts, incumbent router, evaluator-design constants
    assignment.py          A1 selective log and the count-matched A0 log
    evaluators.py          D0 / D1 / D2 distortion streams and the surfaces they induce
    oracles.py             the two labelled oracle controls
    metrics.py             deployment regret, identification error, switch outcomes
    gates.py               held-out decision checks and channel verdict validity
    pipeline.py            one partition of the full design
    validation.py          six leakage and algebra tests, run before any result file is written
  case_core.py             the audit protocol (correct, allocate, re-correct) and baselines
  load_llmrouterbench.py   builds the primary corpus cache from the released tree
  run_tae_validity.py      mandatory package: the 2x2, three budgets, the deployment gates
  run_tae_replication.py   the same design on the second corpus
  run_tae_appendix.py      pre-specified sensitivity sweeps: distortion size and shape, context
                           count, readout size, baseline, and router aggressiveness
  run_tae_correction.py    correction-family ceiling and richer correctors
  run_tae_overlap.py       which cells each acquisition rule and surface actually buys
  run_tae_intervals.py     95% intervals under partition, request and slice resampling
  run_tae_diagnostics.py   reversal-arithmetic check, assignment-invalidity test, gate break-even,
                           equivalence tests
  make_tae_artifacts.py    figures, LaTeX tables, and the registry of every number in the paper
  data/                    where the two benchmark matrices go (see code/data/README.md)
results/                   one JSON per experiment; every number in the paper reads from these
figs/                      the three figures, as produced by make_tae_artifacts.py
docs/
  TAE_ANALYSIS_PLAN.md     the design, outcomes and hypotheses fixed before the confirmatory runs,
                           including the hypothesis that did not survive
  CLAIM_REGISTRY.md        every load-bearing claim mapped to its artifact, result file and the
                           wording the evidence does and does not support
```

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
# place the two benchmark matrices as described in code/data/README.md
cd code
python run_tae_validity.py --partitions 30 --acq-seeds 6   # ~70 s, runs the tests first
python run_tae_replication.py                              # ~60 s, second corpus
python run_tae_appendix.py                                 # ~8 min, sensitivity sweeps
python run_tae_correction.py                               # ~2 min
python run_tae_overlap.py                                  # ~1 min
python run_tae_diagnostics.py                              # ~5 min
python run_tae_intervals.py                                # ~15 min, bootstrap intervals
python make_tae_artifacts.py                               # figures, tables, paper numbers
```

Every script writes to `results/` and prints its own summary. `make_tae_artifacts.py` regenerates
`figs/` and `results/tae_paper_numbers.json`, which holds every value quoted in the paper under a
stable key, so any figure in the text can be checked without rerunning anything. It also writes
`tae_tables.tex`, the LaTeX fragment the manuscript reads; that file belongs to the paper repository
and is ignored here.

Runs are deterministic given the partition and acquisition seeds, so a fresh run reproduces the
committed result files exactly.

---

## Checks that gate the results

`code/eval_validity/validation.py` runs six tests before any result file is written, and the runner
exits non-zero if one fails.

1. **Cohort isolation.** The five request cohorts are pairwise disjoint and exhaustive.
2. **Assignment matching.** A paired A0/A1 run has identical per-cell counts, identical total log
   volume and identical evaluator distortion, so only the assignment distribution differs.
3. **Target-channel leakage.** Replacing the scoring cohort's outcomes with random values leaves the
   acquired cells, the fitted correction, the deployed surface, the candidate policy and the gate
   decision bit-identical. The allocator is called with the scoring surface set to `None`, so reading
   it would raise rather than bias.
4. **Stable-evaluator algebra.** With a shared distortion it cancels from the logged-minus-audit
   difference and survives in the population-operational surface, as the theory requires.
5. **Assignment invariance.** The residual the theory needs to vanish is measured rather than
   assumed, and reported against the selection bias it is meant to identify.
6. **Stream-mismatch algebra.** With a halved audit-stream distortion the estimated difference gains
   the predicted extra term.

---

## Note on labels

The paper writes the evaluator factor as **D** (D0, D1, D2). The code and every result file key the
same factor as **E** (E0, E1, E2, plus E1k and E2rev in the sweeps). The assignment factor A0/A1 is
identical in both. Only the display layer of `make_tae_artifacts.py` renames it. This is recorded in
`docs/CLAIM_REGISTRY.md`.

---

## Data and scope

Neither corpus is redistributed here, and no new model inference is performed anywhere in this
package. Both experiments replay per-request outputs their authors already released: LLMRouterBench,
of whose 21 datasets and 33 models we use five and five, leaving 11,289 requests scored by all five,
and RouterBench, whose zero-shot matrix holds 36,497 requests and 11 models. The evaluator
distortion is a pre-specified synthetic stress test applied to those real outcomes, not a measurement
of any deployed judge, and `docs/CLAIM_REGISTRY.md` records that distinction alongside the wording it
does and does not license.
