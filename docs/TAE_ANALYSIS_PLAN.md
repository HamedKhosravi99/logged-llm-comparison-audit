# TAE analysis plan (fixed before the confirmatory runs)

Everything below was fixed by formula before the result files in `results/tae_validity*.json` were
inspected. Deviations from it, and hypotheses that did **not** survive, are recorded in the last
section — they are reported in the paper as stated here, not silently dropped.

---

## 1. Design

**Data.** LLMRouterBench performance–cost track, offline replay: 11,289 prompts × 5 models across
5 benchmark slices (SimpleQA, MMLU-Pro, HLE, LiveCodeBench, ArenaHard). Five benchmark evaluation
slices sharing one pipeline — *not* five independent studies. No new model inference.

**Grid.** `K = 5` models × `J = 8` frozen prompt-embedding contexts (KMeans fit on router-train
only). A cell is supported when it has ≥ 8 logged prompts and an acquisition pool ≥ the readout size.

**Cohorts** (prompt-disjoint, stratified by slice): router-train 20% / observational log 50% /
acquisition 12% / verification 8% / scoring 10%.

**Factor A (assignment validity).**
* `A1` — FrugalGPT-style cost cascade fit on router-train, exploration rate ε = 0.15.
* `A0` — the *same* log prompts, re-partitioned uniformly at random into blocks of exactly the same
  per-cell counts `n_kj`. Matched log volume, matched per-cell counts, matched evaluator.

**Factor E (measurement validity).** The paper labels this factor D; the code and result files key it E.
* `E0` — `C ≡ 0`.
* `E1` — `C_kj = λ · z_k · s_j` shared by all three streams, where `z_k` is the standardized model
  capability rank and `s_j = sign(d_j − median_j' d_j')` the context-difficulty sign, both from
  router-train only, and `λ = 0.5 × median_j Δ^ref_{j,train}`. Scores clipped to [0, 1].
* `E2` — `C^o = C`, `C^i = ½C`, same-evaluator verification `C^v = C`, reference channel `C^v = 0`.
* `E1k` (appendix) — context-invariant model favouritism `C_kj = C_k = λ·z_k`, the cleanest instance
  of Prop. 1(b): it cancels from the differencing entirely and survives whole in the corrected
  surface.

**Budgets.** 15%, 25%, 35% of supported cells; readout size n = 30 prompts per purchased cell.
Gated arms take `m_v = 2·round(0.30·B/2)` cells *out of* B, so gated and ungated spend the same total.

**Methods.** trust operational evaluation · uniform randomized audit · CASE (corrected margin) ·
R-EPIG on the corrected surface · **population-operational oracle** (`μ^op`, the exact
common-population operational cell means — assignment invalidity removed, evaluator distortion kept)
· reference-outcome oracle (`μ^ref`). Appendix only: raw-margin acquisition, observed-surface
R-EPIG, context-invariant favouritism.

**Gates.** ungated · same-evaluator held-out decision check · reference-channel held-out decision
check. The status quo is the **log-derived deployment policy** `argmax_k M_kj` — the model-selection
conclusion the operational log supports today, *not* the incumbent router that generated the log
(that is an appendix variant, and it makes the comparison uninformative).

---

## 2. Outcomes

**Confirmatory primaries (two, no more).**

1. Reference-channel deployment regret
   `R^ref(π̂) = (1/J) Σ_j [ μ^ref_{π^ref(j),j} − μ^ref_{π̂(j),j} ]` on the untouched scoring cohort;
   the same formula restricted to each slice's scoring prompts, with the pooled target policy fixed.
2. False-switch rate: the gate replaces the log-derived deployment policy with the candidate **and**
   the candidate has strictly lower reference-channel value on the untouched scoring cohort.

**Secondary, descriptive only.** top-1 identification error · logged ranking-reversal rate ·
false-retention rate · switch rate · operational-channel regret · channel verdict-inversion rate.

**Decomposition quantities.**
`G_sel = R(trust) − R(population-operational oracle)` ·
`G_meas = R(population-operational oracle) − R(ref. oracle)` ·
`F_repair = (R(trust) − R(CASE)) / (R(trust) − R(population-operational oracle))`, reported only
where the denominator exceeds 1e-4 · `G_construct = R(CASE | E1) − R(CASE | E0)`.

`F_repair` is reported **two ways**, because they are different estimands: the mean over partitions
of the per-partition ratio (24% under A1–E1) and the ratio computed from the aggregate means (30%).
The paper quotes both rather than letting them be confused.

---

## 3. Replication and inference

Replication unit is the **prompt-cohort partition**: 30 partitions × 6 acquisition seeds = 180 runs
per design cell. Acquisition seeds are averaged *within* a partition before any interval is formed;
every interval is a paired standard error over the 30 partitions. Partitions and seeds are identical
across methods. Seed-level standard errors are a simulation-stability measure only.

**Intervals.** The default `±` is a paired standard error over partitions. `run_tae_intervals.py`
reports 95% CIs for every confirmatory contrast under three resampling schemes, because they answer
different questions: **partition** (paired t + percentile bootstrap; would a different split and
acquisition draw agree?), **prompt** (nonparametric bootstrap of the scoring cohort, policies fixed,
μ^ref and π^ref re-estimated per replicate; is the answer an artifact of which prompts were scored?),
and **slice** (cluster bootstrap over the five benchmark slices; would it survive a different
suite — coarse by construction with five clusters).

**Pre-specified primary comparisons (four).**
1. A1–E0: CASE vs. trust, reference regret.
2. A1–E0: R-EPIG on the corrected surface vs. CASE (tests non-uniqueness of the acquisition rule).
3. A1–E1: CASE vs. the population-operational oracle and the reference-outcome oracle.
4. A1–E1 / A1–E2: same-evaluator vs. reference-channel false-switch rate.

Non-significance is never reported as equivalence. A small measurement floor is reported as small.

---

## 4. Hypotheses, and what happened

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Under A1–E0, `R(CASE) < R(trust)` on the macro average and a majority of slices | **Supported** (−0.0161 ± 0.0028; 27/30 partitions; 5/5 slices). Partial, not near-complete: ≈32% of the gap to the population-operational oracle. |
| H2 | Under A0–E1, CASE does not close the gap to the reference oracle | **Supported** (CASE 0.0330 vs. population-operational oracle 0.0030; CASE is worse than trust) |
| H3 | Under A1–E1, partial repair with a non-zero measurement floor | **Supported** (`F_repair` = 24%, `G_meas` = 0.0030 ± 0.0006, top-1 floor 18.3%) |
| H4 | Under E1, same-evaluator false switches > reference-channel false switches | **NOT supported.** Paired difference −0.011 ± 0.008 (A1–E1) and −0.028 ± 0.012 (A1–E2): the same-evaluator check makes *fewer* harmful switches, bought with retentions. Reported as stated, with the verdict-inversion measurement added to show what the check *cannot* do. |
| H5 | Under A1–E2, CASE moves away from the population-operational oracle | **Inconclusive at the tested magnitude** (+0.0010 ± 0.0012 at 35%; +0.0022 ± 0.0010 at 25%) |

**Pre-specified sensitivities** (run only after the mandatory package, `run_tae_appendix.py`):
λ = 1.0 × median training margin · sign-reversed stream mismatch · readout size n ∈ {15, 30, 60} ·
incumbent-router status quo · context-invariant model favouritism `C_kj = C_k` · raw-margin and
observed-surface acquisition.

**Deviations from the plan.** Two additions, both labelled as diagnostics rather than primaries:

1. The *channel verdict inversion rate* — the fraction of contested contexts where the checking
   channel's population-level comparison disagrees with the target channel's — was added after H4
   failed, to characterise what a same-evaluator check cannot do at any budget.
2. The *acquisition-overlap* measurement (`code/run_tae_overlap.py`) was added after review asked
   what CASE contributes if it only ties R-EPIG on the corrected surface. Equal regret is weak
   evidence that a rule does not matter, so the cell sets each arm purchases are compared directly:
   on the corrected surface the two rules buy 86.5% of the same cells, against 61.4% when the same
   margin rule is moved to the observed surface. The tie is mechanical, and R-EPIG-on-the-corrected-
   surface is an ablation of the protocol rather than a competing method (the published R-EPIG
   acquires on the observed surface).
3. The *correction-family ceiling* (`code/run_tae_correction.py`) was added after review pointed out
   that Prop. 1 is cellwise while the deployed corrector is action-level, so the incomplete recovery
   could be misspecification rather than budget. It is the latter: the best-fitting member of the
   action-level family, computed from the truth with no budget, reaches 0.0318 against CASE's 0.0338
   at a 35% budget (paired +0.0020 ± 0.0024). Of the trust-to-oracle gap, 28% is repaired, 5% remains
   reachable inside the family, and 66% lies outside it. Richer correctors (two-way, rank-2,
   experiment-grounded shrinkage) do not help at this budget.
4. *95% confidence intervals under three resampling schemes* (`code/run_tae_intervals.py`) were
   added after review objected that a partition SE is not uncertainty over future prompts or
   benchmark suites. Consequence, reported in the paper: the audit benefit and the CASE/R-EPIG tie
   hold under all three schemes, but the measurement floor *in value* (0.0030) and the cost of
   auditing under A0 (+0.0107) both lose separation from zero under the five-slice cluster
   bootstrap. The identification-side floor (18.3%) is unaffected.
5. The *context-granularity sweep* J ∈ {6, 8, 12, 16} was added at the same time. Every qualitative
   conclusion holds at every J; the identification floor ranges 10.4–18.3%, and the pre-specified
   J = 8 happens to give the largest of the four, so 18.3% is the top of a range, not a constant.
6. A *replication on a second corpus* (`code/run_tae_replication.py`, RouterBench: 36,497 requests,
   11 models) was added after review identified one corpus as the paper's biggest exposure. It
   changed two claims: the acquisition-rule tie and the cost of auditing under a common-population
   log are both corpus-specific and are now reported as such.
7. Four *sceptic's diagnostics* (`code/run_tae_diagnostics.py`): whether the reversal rate is
   arithmetic (largely yes — predicted 29.0% vs observed 18.5% from margins alone), a usable
   assignment-invalidity test, the gate's break-even loss asymmetry, and equivalence tests (TOST)
   for the two claims previously stated as ties.
8. The *assignment-invariance residual* `|C^{o,sel} − C^{i,pop}|` was added after review pointed out
   that Prop. 1's cancellation needs the evaluator's mean error to be invariant to routing within a
   context, which is strictly stronger than "the same judge scored both streams". Clipping to [0,1]
   makes the realised error mildly prompt-dependent, so the residual is measured rather than assumed:
   0.0018 ± 0.0001, against a 0.0013 cohort-sampling floor and a selection bias of 0.0991 — 1.8%
   contamination. The context-invariant-favouritism sweep (`C_kj = C_k`) was added at the same time
   to show the measurement result is not an artefact of the capability × difficulty interaction.

---

## 5. Leakage and validity tests

`code/eval_validity/validation.py` runs all of these before any result file is written; the runner
exits non-zero if one fails. Cohort isolation · assignment matching (per-cell counts, total volume,
evaluator) · reference-channel leakage (scramble the scoring outcomes; nothing downstream may move) ·
stable-evaluator algebra · assignment invariance · stream-mismatch algebra. The allocator is called
with the scoring surface set to `None`, so reading it would raise rather than bias.

---

## 6. Claim language

**Supported.** "CASE repairs a substantial fraction of router-induced assignment bias under the
stated identification and correction-fit conditions." · "Correcting the surface before acquisition is the operative evaluation
principle." · "A stable shared evaluator bias remains after assignment correction." · "The experiment
distinguishes assignment validity from measurement validity relative to a designated target outcome."
· "Corrected-surface R-EPIG matches CASE."

**Prohibited.** "CASE repairs *most* of the assignment failure" (it closes ≈30% of the gap to an
oracle repair) · "the same judge on both streams makes the distortion cancel" (assignment invariance
is the actual condition) · "CASE guarantees trustworthy evaluation." · "CASE corrects evaluator bias." · "The
benchmark is ground truth." · "The policy-value gate certifies safe deployment." · "CASE uniquely
dominates active evaluation." · "Reference-channel verification is generally available in
production." · "Five datasets constitute five independent studies." · "A non-significant difference
proves equivalence."
