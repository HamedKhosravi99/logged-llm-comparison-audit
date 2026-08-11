# Predeclared plan: Phase-0 analyses for the revision

FROZEN before any Phase-0 outcome is computed. Same discipline as `KUAIRAND_ANALYSIS_PLAN.md` and
`CRITEO_ANALYSIS_PLAN.md`. Deviations require a dated entry in the log at the bottom and must be
reported wherever the result is quoted.

Order (set by reviewer demand, not convenience): **P0.5 → P0.2 → P0.1**, then P0.3, P0.4, P0.6,
P0.7, P0.8. Label validity is upstream of everything measured in that label.

---

## P0.5 — Resolution of the "true domain best" label

**Question.** Table 2 classifies every switch against the true domain best, computed as
`tc.groupby(["domain","agent_model"]).success.mean()` over ~440 τ²-bench simulations. With ~37
simulations per (model, domain) cell, is that label resolved at the tolerance δ = 0.02 it adjudicates?

**Procedure.**
1. Per (domain, model): success rate, n, and a Wilson 95% interval.
2. Per domain: the argmax model, and the set of models whose interval overlaps the argmax's interval.
3. A domain decision for candidate `cand` is **indeterminate** if the sign of
   `TRUE[cand] − (best − δ)` is not determined at the cell intervals — i.e. if the interval for
   `TRUE[cand] − best` straddles `−δ`.
4. Bootstrap: resample simulations within (domain, model) 1,000×, recompute the label, report the
   share of resamples in which each (domain, candidate) classification flips.

**Predeclared trigger.** Let *I* = share of domain decisions with an indeterminate label.
- I > 25% → Table 2's taxonomy is reported as **indicative only**; no confirmatory claim rests on it.
- 10% ≤ I ≤ 25% → report Table 2 as a **bracketed range** over the most- and least-favorable
  resolution of indeterminate labels.
- I < 10% → proceed as published, with the count stated in §6.5.

**Reported regardless of outcome:** the ~12 (model, domain) estimates with intervals; the integer
count of indeterminate decisions; and the already-established fact that all 10 valid cost-saving
opportunities are in `telecom`.

**Artifact:** `results/label_resolution.json`.

## P0.2 — Null for the KuaiRand arm-share decomposition

**Question.** Is the 79.6% arm-main-effect share of `C_kj = click_kj − like_kj` distinguishable from
what heterogeneous per-arm base rates produce mechanically?

**Null (corrected; v2's within-cell pairing permutation was degenerate and is discarded).** Permute
**arm labels across items within a context**, preserving each context's item composition and both
signals' base rates, breaking only the arm↔content association. 1,000 permutations.

**Statistic.** The arm share of the two-way additive fit, exactly as `c_decomp` computes it.

**Predeclared reading, two-sided.** Removing true interaction *raises* the arm share, so a one-sided
"below the 95th percentile" rule is uninformative. Report where 79.6% falls in the null distribution.
- 79.6% at or below the null's median → the decomposition carries no information beyond arm base
  rates; **withdraw the claim** from the abstract and report it as descriptive only.
- 79.6% above the null's 95th percentile → the claim stands as more than a base-rate artifact.
- Between → report the percentile and state that the evidence is weak.

**Positive control (direction corrected).** Inject a known *context-varying* $C_{kj}$ at the realized
cell sizes and confirm the fit attributes it to interaction. If it cannot, then "79.6% is arm main
effects" only means the estimator cannot resolve interaction at these cell sizes, and that is
reported as the finding.

**Additional diagnostic, run regardless (demanded independently by two reviewers).** The arm-main-
effect share of the **click table alone** and the **like table alone**, beside the share for their
difference. If click alone shows a comparable arm share, the result is about click heterogeneity,
not about evaluator error.

**Artifact:** `results/kuairand_share_null.json`.

## P0.1 — Bootstrap interval for the arm share

**Procedure.** Cluster bootstrap over **users**, 1,000 replicates, **refitting the two-way
decomposition inside every replicate**. Both pools reported: scoring (predeclared) and full stream.

**Known bias, stated in advance.** With one $C$ value per cell, the residual absorbs binomial noise,
so the plug-in arm share is **downward**-biased; the nonparametric bootstrap adds a second dose of
the same noise. The interval is therefore conservative for the arm share and must not be read as if
unbiased.

**Predeclared trigger.** Governing quantity: the **lower bound on the scoring pool**.
- LB ≥ 60% → keep in the abstract with the interval attached.
- LB < 50% → **delete from the abstract**; keep descriptively in App. I.
- 50% ≤ LB < 60% → App. I only, with the interval; no abstract number.
87.6% (full stream) is never quoted without 79.6%.

**Artifact:** `results/kuairand_share_bootstrap.json`.

---

## Verification requirement (all Phase-0 items)

Every number that reaches the paper must be recomputed on a second, independently written code path
before it is quoted, as was done for KuaiRand and Criteo. Any procedure must first be shown
non-degenerate on a worked example — the standing rule added after v2's permutation null was found
to leave the test statistic exactly invariant.

## Deviations log

- **2026-08-10, P0.5 second-path disagreement (recorded, not resolved by choosing).** The
  predeclared bootstrap gives 5/12 indeterminate (41.7%, trigger INDICATIVE_ONLY). An independently
  written analytic path (paired per-task differences against a *fixed* argmax) gives 3/12 (25.0%).
  They differ by design: the analytic method conditions on the argmax being correct and therefore
  can never flag the argmax itself, while the bootstrap re-estimates the argmax in every replicate.
  Since the label *is* an estimated argmax, the bootstrap is the faithful operationalization and the
  analytic 25% is a lower bound. Both are reported; the trigger fires on either reading
  (>25% = indicative only; 25% = the boundary of the bracketed-range branch).
- **2026-08-10, sample-size unit.** The paper's "every model--domain cell at $n\ge24$" counts raw
  simulations. 181 of 239 (domain, task, model) triples were run twice, so the independent unit
  count is 19--20 **tasks** per cell. All P0.5 analyses cluster on tasks.
