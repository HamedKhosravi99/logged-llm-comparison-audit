# Can We Trust Logs to Choose a Cheaper LLM? — reproducibility package

Code, result files, figures and analysis plans for the workshop paper *Can We Trust Logs to Choose a
Cheaper LLM? Auditing Selection and Measurement Validity in Multi-Model Deployments*, targeted for
the TAE workshop at NeurIPS 2026, *Can We Trust AI Evaluation? Robustness, Causality, and Risk in
Modern AI Assessment* (<https://tai-eval.github.io/>).

**TLDR.** A routed production log can support the wrong cheaper-model decision for two unrelated
reasons, and a randomized audit repairs exactly one of them. This package runs the crossed experiment
that separates the two, prices the decision in dollars, and releases the tool that applies it to your
own traces.

---

## RouterAudit — apply this to your own router

**RouterAudit** is the tool the paper releases. Point it at a trace export from your own router and
it returns a verdict on whether the log can support the decision, a corrected comparison, a list of
the comparisons worth buying, and the traffic volume at which the evaluation pays for itself. It runs
offline: it never calls a model API and never changes production routing.

```bash
python code/router_audit.py validate  traces.csv                     # can this log answer it at all?
python code/router_audit.py diagnose  traces.csv                     # is it imbalanced enough to audit?
python code/router_audit.py recommend traces.csv --budget-usd 500    # what to buy next
python code/router_audit.py simulate  traces.csv --quality-tolerance 0.01
python code/router_audit.py report    traces.csv --out report.md     # the decision record
```

Required columns are `request_type`, `candidate_model` and `automatic_score`; five optional columns
(`inference_cost_usd`, `traffic_weight`, `target_score`, `selected_model`, `request_id`) each unlock
more of the output, and an OpenTelemetry/Phoenix span export maps onto the schema directly.

The step that saves money is `diagnose`: **auditing an already balanced log makes the decision
worse**, and the tool will tell you to keep your money.

**Full practitioner walkthrough — how to read the output, and what the tool refuses to do:
[`docs/ROUTERAUDIT.md`](docs/ROUTERAUDIT.md).** A browser demo of the same tool is in
[`routeraudit-space/`](routeraudit-space/).

---

## The two failures the paper separates

A production router sends easy requests to cheap models and hard requests to strong ones, then logs
an automatic score for each answer. Reading a model comparison off that log can fail because

* **assignment validity** fails, since the models were scored on different requests, and
* **measurement validity** fails, since the automatic scorer may not measure the outcome the
  deployment decision is about.

The experiment switches each failure on and off independently on the same data, holding the model
roster, requests, contexts and log volume fixed.

|                                | faithful evaluator (**D0**) | distorted evaluator (**D1**) |
|--------------------------------|-----------------------------|------------------------------|
| **common-population log (A0)** | no systematic threat        | measurement only             |
| **router-selected log (A1)**   | assignment only             | both                         |

## Headline results

At a 35% randomized-cell budget on the primary corpus unless stated otherwise; `±` is a paired
standard error over 30 partitions.

**On a real production log (KuaiRand-Pure), repairing assignment made the decision worse.** The log
carries a selectively assigned and a randomized stream over the same users and window, with the same
cheap signal (`is_click`) and the same designated outcome (`is_like`) on every row, so all four
validity cells exist with nothing simulated.

* A **cellwise-perfect assignment repair leaves more loss (2.01) than trusting the log unrepaired
  (1.76)**, and more than the audit protocol (1.57): the penalty is +0.44 [+0.32, +0.56] against
  CASE and +0.26 [−0.02, +0.54] against trust-the-log, in `is_like` units ×10⁻³. Two biases were
  partly cancelling, and removing one exposed the other.
* Evaluator error against the platform's own designated outcome is **79.6%** arm main effects, 9.1%
  request-type, 11.3% residual — exactly the audit-invisible form the theory predicts. A user-level
  cluster bootstrap gives [57.9, 78.2]%; permuting arm labels within request type puts the observed
  share above all 1,000 permutations.
* Ranking arms by **traffic share** picks a target-best option in **none** of the eight request types.
* On the Criteo encouragement RCT, **79%** of the measurement gap survives instrumenting.

On the controlled corpora:

* The routed log names the wrong cheapest-good-enough model in **70%** of request types. On the
  stricter top-1 criterion the failure is mostly a sampling floor: at matched per-pair counts the
  selective log errs on 67.5%±2.4 against 39.6%±2.5 for a comparable-request log.
* Auditing repairs assignment where the log is imbalanced — target-value loss falls from
  0.0499±0.0030 to **0.0338±0.0020** — and *hurts* where it is not, raising loss by +0.0107.
* No budget buys measurement validity: with selection removed exactly, the evaluator still picks the
  wrong model in **18.3%±2.7** of request types, and two real judges on the same battles agree with
  the human verdict on 0.444 and 0.300 with length preferences of **opposite sign**.
* The binding limit is the correction family, not the budget: **68.4%±2.0** of the selection bias is
  not row-constant. The uncapped best fit in that family sits [−0.0030, +0.0070] from deployed CASE
  — two one-sided tests do *not* establish equivalence (p = 0.11), so this is a tie the study is not
  powered to resolve, not a demonstrated one.
* Valid opportunities are rare. In the operational replay of 440 verified customer-service
  simulations, a quality-preserving cheaper alternative exists in only **6.7%** of decisions and 80%
  of runs contain none.
* The audit is cheap ($2.39±0.08 at released benchmark prices), but **auditing by itself raises
  deployment spend**, from $3,358 to $5,891 per million: it moves traffic toward better and dearer
  models. The quality *tolerance* is what saves. Together they save $1,579 per million
  [1,019, 2,227], repaying the audit after **1,514 requests** [1,064, 2,368]. All dollar figures are
  estimated counterfactual production costs computed from released benchmark prices, not any
  company's observed spend.

Every number quoted in the paper is registered in `results/tae_paper_numbers.json` under a stable
key, so any figure in the text can be checked without rerunning anything.

## Layout

```
code/
  eval_validity/          the crossed validity experiment package
  case_core.py            the audit protocol (correct, allocate, re-correct) and the baselines
  run_tae_*.py            the confirmatory replay, its replication, sweeps and intervals
  run_causal_*.py         propensity feasibility gate and the IPW / DR / regression baselines
  econ_core.py, run_econ.py, free_stage1_*.py, free_stage2_*.py
                          the economic and operational analyses
  tau2_*.py               the 440-simulation customer-service replay
  lmarena_*.py, helpsteer_run.py, run_lmarena_share_test.py
                          the measured-judge study on human-labelled battles, and the arm-share
                          decomposition on a balanced model-by-category grid
  kuairand_download.py, run_kuairand_validity.py, run_kuairand_share_tests.py
                          the production-log study: the protocol on real traffic, and the
                          bootstrap / permutation tests behind the arm-level share
  run_criteo_encouragement.py, verify_criteo_encouragement.py
                          the encouragement design on a real randomized experiment
  run_gate_vs_audit.py, run_kappa_transfer.py, run_label_resolution.py, verify_label_resolution.py
                          the pre-registered Phase-0 checks (gate vs audit, diagnostic transfer,
                          label resolution)
  verify_prop_idset.py    brute-force check of the point-identification proposition against the
                          achievable winner sets, on a lattice that realizes boundary cases
  verify_mckp_certificate.py
                          brute-force check of the robust-optimality certificate for budgeted
                          allocation, against rival-by-rival worst cases over the coefficient box
  router_audit.py         RouterAudit
  make_*.py               every figure and table in the paper, regenerated from result files
  data/                   where the two benchmark matrices go (see code/data/README.md)
results/                  one file per experiment; every number in the paper reads from these
figures/                  every figure in the paper
docs/
  TAE_ANALYSIS_PLAN.md    design, outcomes and hypotheses fixed before the confirmatory runs
  ANALYSIS_PLAN.md        the same, for the economic revision
  KUAIRAND_ANALYSIS_PLAN.md, CRITEO_ANALYSIS_PLAN.md, PHASE0_ANALYSIS_PLAN.md
                          the same, each frozen before any outcome column was read
  CRITEO_ENCOURAGEMENT_RESULTS.md
                          what the encouragement study returned, against that frozen plan
  CLAIM_REGISTRY.md       every claim mapped to its artifact, result file, and the wording the
                          evidence does and does not support
  CLAIM_BOUNDARIES.md     the wordings the evidence forbids, and the required companions
  ROUTERAUDIT.md          how to apply RouterAudit to your own traces
routeraudit-space/        browser demo of RouterAudit
```

A submission-verification script that recomputes every table and figure number and checks the
manuscript is withheld from this package: it names the authors, so including it would defeat the
double blind.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
# place the two benchmark matrices as described in code/data/README.md

python code/run_tae_validity.py --partitions 30 --acq-seeds 6   # ~70 s, runs the design tests first
python code/run_tae_replication.py                              # ~60 s, second corpus
python code/run_tae_appendix.py                                 # ~8 min, sensitivity sweeps
python code/run_tae_correction.py                               # ~2 min, correction-family ceiling
python code/run_tae_overlap.py                                  # ~1 min
python code/run_tae_diagnostics.py                              # ~5 min
python code/run_tae_intervals.py                                # ~15 min, bootstrap intervals

python code/run_causal_feasibility_gate.py                      # propensities, before any estimator
python code/run_causal_baselines.py --partitions 30             # ~40 s, IPW / DR / regression

# the real-data studies (each needs its own public dataset; see the matching plan in docs/)
python code/kuairand_download.py                                # KuaiRand-Pure, Zenodo 10439422
python code/run_kuairand_validity.py --data code/data/kuairand  # the protocol on production traffic
python code/run_kuairand_share_tests.py --data code/data/kuairand   # bootstrap + permutation null
python code/run_criteo_encouragement.py --data <criteo-uplift>  # the encouragement design

# pre-registered Phase-0 checks
python code/run_gate_vs_audit.py
python code/run_kappa_transfer.py
python code/run_label_resolution.py

python code/make_tae_artifacts.py                               # figures, tables, paper numbers
python code/make_fig1_validity.py \
  --validity results/tae_validity.json \
  --paper-numbers results/tae_paper_numbers.json \
  --output figures/fig1_validity.pdf
python code/make_fig_allocation.py --outdir figures            # the illustrative allocation figure
```

Runs are deterministic given the partition and acquisition seeds, so a fresh run reproduces the
committed result files exactly. `code/eval_validity/validation.py` runs six design tests — cohort
isolation, assignment matching, target-channel leakage, stable-evaluator algebra, assignment
invariance, stream-mismatch algebra — before any result file is written, and the runner exits
non-zero if one fails.

## Data and scope

No dataset is redistributed here; every study replays data its authors already released, and each
loader points at the public source.

* **LLMRouterBench** (11,289 requests scored by all five models used) and **RouterBench** (36,497
  requests × 11 models) — the controlled validity experiments. See `code/data/README.md`.
* **KuaiRand-Pure** (Zenodo 10439422, CC BY 4.0) — the production log with a built-in randomized
  arm. `code/kuairand_download.py` fetches it.
* **Criteo-UPLIFT v2.1** — the randomized encouragement experiment.
* **LMArena** battles and **HelpSteer2** — the measured-judge study, on human labels.

The synthetic evaluator distortion is a pre-specified stress test applied to real outcomes, not a
measurement of any deployed judge; the measured-judge results come separately from the human-labelled
corpora above. On KuaiRand the designated outcome (`is_like`) is a platform-chosen target, not
independent welfare, and there is no price column, so none of the paper's dollar results extend
there. `docs/CLAIM_REGISTRY.md` records these distinctions alongside the wording each does and does
not license, and `docs/CLAIM_BOUNDARIES.md` records the wordings the evidence forbids.
