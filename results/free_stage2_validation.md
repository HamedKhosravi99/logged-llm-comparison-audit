# Stage-2 validation memo: incumbent-anchored paired audit for the cost-reduction decision

Status: **post-hoc exploratory protocol analysis.** The redesign was developed after inspecting
stage-1 results, and every number below comes from the same 440 tau2-Bench-Verified simulations
(50 fold x split-seed units) used to diagnose the failure. Nothing here is confirmatory or
pre-specified. No paper edits have been made.

Code: `free_stage2_validation.py` (all sections), `free_stage2_diagnosis.py`,
`free_stage2_variants.py`. Machine-readable outputs:
`free_stage2_validation_summary.csv`, `free_stage2_target_table.parquet`,
`free_stage2_validation_units.parquet`, `free_stage2_cost_accounting.parquet`,
`free_stage2_calibration.csv`, `free_stage2_domain_table.csv`,
`free_stage2_frontier.parquet`, `free_stage2_sensitivity.csv`.

This memo supersedes the interpretive claims in `free_stage2_protocol_analysis.md`
(two corrections are flagged in §12 and §8).

---

## 1. Decision target: the gate does NOT certify the paper's objective

The paper's rule accepts a cheaper model when `q_cand >= q_best - delta`. The new gate tests
`q_cand >= q_incumbent - delta`. These differ whenever the incumbent is not the best:

- Incumbent IS the true best model: **40.0%** of the 150 domain-unit cells (population truth);
  **48.7%** on held-out scoring-split estimates.
- Mean incumbent-to-best gap **+0.088**, maximum **+0.425** (telecom).

The correct name is therefore **incumbent-anchored non-inferiority gate**: it controls
degradation relative to the deployed policy and does not certify the best-relative
cheapest-good-enough condition.

Dual classification of switches per run (best-relative A/B/C/D uses the paper's definition;
incumbent-relative uses cheaper+non-inferior / cheaper+inferior / no-economic-saving):

| arm | best-rel A | B | C | D | inc-rel valid | inferior | no-econ |
|---|---|---|---|---|---|---|---|
| Raw-log downrouting | 0.040 | 0.200 | 0.540 | 0 | 0.200 | 0.380 | 0.200 |
| Stage-1 best-relative gate (S0) | 0.064 | 0 | 0.091 | 0 | 0.092 | 0.063 | 0 |
| Locked: paired NI tau=0.95 | 0.140 | 0 | 0.120 | 0 | 0.220 | 0.040 | 0 |
| Locked + Bonferroni | 0.040 | 0 | 0.020 | 0 | 0.060 | 0.000 | 0 |

Category C is NOT reinterpreted as valid. The locked arm's best-relative C = 0.120 stands; its
structural source is §10 (airline: gpt-4o-mini is within delta of the incumbent but 0.042
below the best). Both classifications must appear wherever switches are reported.

If incumbent-relative non-inferiority became the primary deployment objective, these paper
elements would need explicit revision (list in §14): the abstract's decision definition, §2
decision rule and its "ideal target decision" sentence, contribution 3 and 4, §5 gate
paragraph, RQ5, Table 1, Figure 2, the Conclusion, and App. algorithm box. No theorem changes
are required if the best-relative objective stays primary and the gate is reported as a
non-inferiority device; a parallel incumbent-relative definition would be needed otherwise.

## 2. What outcome the gate uses

The gate scores both models with the stored **tau2-Bench-Verified `success`** field: the
paper's designated target outcome, bounded in [0,1] with support {0, 0.5, 0.667, 1} — NOT an
imperfect automatic evaluator, and not binary. Consequences, stated plainly:

- This is a **measurement-validated audit channel**: the replay certifies non-inferiority in
  the designated target outcome because the benchmark stores verified outcomes for every
  (task, model) cell. This matches the stage-1 RQ5 pipeline (same field), so no comparison in
  this memo mixes channels.
- In a real deployment the verified outcome costs labeling/verification labor per audited
  task. The recorded audit cost covers **model-call cost only**; label cost is excluded and
  must be stated as such in any paper text (the Limitations paragraph already notes label
  cost; the protocol description must add it explicitly).
- Claims must remain channel-relative: "non-inferiority in verified task success," not
  "target-quality preservation" in general.

## 3. Audit-cost accounting

Per-unit identity: `audit_usd(unit) = sum over audited (domain, model) of stored per-task
model-call cost on the full domain pool`. Verified at call level:

- identity **HOLDS** in all 50 units; mean $0.5296 = reported $0.5296.
- Composition over 50 units: 290 candidate calls (862 tasks, $10.54 total) + 150 incumbent
  calls (450 tasks, $15.94 total). The incumbent re-scoring is 60% of spend; the log-anchored
  variant that skips it costs $0.2108/unit.
- Included: all candidate calls (also failed investigations and domains where no switch was
  made), all incumbent calls, all audited domains. Excluded: label/verification cost (see
  §2); screening calls (none exist — candidates come from the price table).

## 4. Small-sample inference and calibration

Model: outcome bounded [0,1]; paired estimator `dbar = mean(cand_i − inc_i)` on common audit
tasks; variance = sample SE of the paired differences, **floored at sqrt(0.25/n)** (the [0,1]
worst case) so all-identical tiny samples never read as certainty (a 3-task tie gives
p = 0.53 → retain); no prior (frequentist plug-in normal); **pass iff the one-sided 95% lower
confidence bound of dbar exceeds −delta**, equivalently Phi((dbar+delta)/SE) >= 0.95. This is
a one-sided non-inferiority bound, not a two-sided interval.

Empirical calibration (paired-outcome distribution resampled from the corpus, 20k reps;
false-switch = pass when true gap <= −delta):

| true gap | n=3 | n=4 | n=5 |
|---|---|---|---|
| −0.10 (clearly inferior) | 0.024 | 0.032 | 0.024 |
| −0.02 (boundary) | **0.076** | 0.041 | 0.044 |
| 0.00 | 0.075 | 0.045 | 0.055 |
| +0.10 | 0.074 | 0.099 | 0.125 |
| +0.425 (telecom-size gap) | 0.334 | 0.495 | 0.481 |

- At n=3 the test is **mildly anti-conservative at the boundary (7.6% vs nominal 5%)**;
  n=4-5 are near nominal. Bonferroni over 3 candidates gives 1.3-2.3% at the boundary.
- Exact enumeration of the actual telecom pair (all C(20,3) subsets): power 0.540 at the true
  +0.425 gap — which fully explains the 3 missed opportunities — and size 0.018 at the
  boundary.
- The gate margin sweep (0 to 0.05) changes nothing at n=3: decisions are dominated by the
  discrete outcome pattern (essentially all-wins is needed).
- Recommendation for the confirmatory stage: audit pools of n >= 5 per domain, or an exact
  test; report the n=3 anti-conservatism wherever the replay numbers appear.

## 5. Candidate selection and multiple testing

- Candidates per domain: 1-3 (mean 1.93), generated from the price table only (models
  strictly cheaper than the incumbent). Acquisition is NOT adaptive; no interim stopping; the
  audit is one fixed-size pass over the full pool.
- The SAME 3-5 audit tasks test all candidates and the cheapest passer is selected —
  a post-selection concern. Per-candidate nominal one-sided level 5%; family-wise bound
  k x 5% <= 15%.
- Empirical policy-level false-switch rate (switch inferior to the incumbent beyond delta,
  per domain-decision): locked 0.0133, Bonferroni 0.0000, stage-1 gate 0.0210, raw-log
  downrouting 0.1267.
- The Bonferroni variant controls FWER but collapses detection at this pool size: valid
  switches 0.140 → 0.040, saving vs incumbent +1.41 [−0.55, +3.37] — no longer supported.
- For the confirmatory stage, pre-specify one of: fixed-sequence testing cheapest-first,
  split selection/confirmation pools, or simultaneous bounds. On this replay we report both
  unadjusted and Bonferroni results transparently and claim only the unadjusted arm WITH its
  empirical 1.3% inferior-switch rate alongside.

## 6. Leakage checks (assertion output)

- PASS audit/scoring/router/log splits pairwise disjoint in all 50 units (same request never
  in audit and held-out evaluation).
- PASS incumbent chosen from router-split data only.
- PASS gates read audit-pool outcomes only; the truth table is never passed to a deployable
  arm (enforced by function signatures; TRUE is used only for post-hoc classification after
  policies are fixed).
- PASS held-out scoring split used once per arm, after the policy is fixed.
- NOTE the price table is the mean stored per-request cost over all tasks — the paper's
  known-API-price assumption. Re-estimating prices without the scoring split leaves every
  candidate set unchanged (100.0% agreement).
- NOTE no verification outcome influences candidate generation, thresholds, or stopping;
  thresholds were fixed (tau = 0.95 primary) before the validation run.

## 7. Exploratory status and the locked protocol

Locked stage-2 protocol (fixed, to be evaluated on independent data):
candidates = models strictly cheaper than the incumbent (price table); audit = candidate and
incumbent on the full per-domain audit pool; test = one-sided 95% non-inferiority with margin
delta = 0.02, SE floored at sqrt(0.25/n); selection = cheapest passer, ties by (price, name);
no early stopping; maximum audit size = the domain pool; cost = every candidate + incumbent
call on the pool, all domains.

Independent evaluation options, in order of preference: (1) a cost-reduction replay on
RouterBench / LLMRouterBench complete matrices (present in the repo and NOT used to design
stage 2); (2) a new tau2 simulation batch (requires paid API calls). Until one is run, all
stage-2 numbers are exploratory and inferential language must stay soft.

## 8. Both economic comparators (both stay in the main results)

- Comparator 1 (current incumbent): *does adopting the audited policy reduce the bill
  relative to doing nothing?* — **Relative to the current incumbent, the targeted paired
  audit has positive estimated savings and a positive 95% lower bound**: +$5.98/1k tasks,
  95% CI [+1.91, +10.05]; lower-bound break-even relative to the incumbent ≈ 187 tasks
  (point ≈ 89) at $0.53 audit spend.
- Comparator 2 (free raw-log downrouting): *does purchasing the audit add value beyond a free
  log-based policy?* — **Relative to free raw-log downrouting, the incremental saving remains
  uncertain because the confidence interval crosses zero**: +$3.77/1k, CI [−0.94, +8.48].

Correction to the earlier stage-2 analysis file: the stage-1 no-payback conclusion was a
result against free raw-log downrouting and **stands unchanged**; it was not an artifact. What
stage 2 changes is detection and the incumbent-comparator estimate. The word "certified" is
retired; use "positive 95% lower-bound estimated saving" / "statistically supported saving
relative to the incumbent."

## 9. Fair free baselines and the cost-vs-violation frontier

No-audit baselines on identical units (log evidence only):

| free policy | cost/1k | switches | best-rel C | inc-rel inferior |
|---|---|---|---|---|
| log NI gate tau=0.95 | 35.14 | 0.000 | 0.000 | 0.000 |
| log NI gate tau=0.9 | 34.91 | 0.100 | 0.100 | 0.060 |
| log NI gate tau=0.8 | 32.17 | 0.300 | 0.280 | 0.220 |
| log NI gate tau=0.5 | 28.19 | 0.600 | 0.540 | 0.380 |
| raw-log margin sweep 0→0.3 | 32.93→29.98 | 0.78→0.94 | switch-C 0.54+ | 0.38→0.58 |
| **locked audited arm** | **29.16** | 0.26 | 0.120 | 0.040 |

The pattern: with only 1-3 exploration observations per cheaper model, **free log evidence
cannot pass any conservative gate** — the safe free policies barely switch and save nothing —
while aggressive free policies reach similar cost only with 5-10x the inferior-switch rate.
At matched conservatism the audit's value is supported: locked vs free log-gate tau=0.9
(inferior rates 0.040 vs 0.060) saves **+$5.74/1k [ +1.64, +9.85]**; vs tau=0.95 (never
switches) +$5.98 [+1.91, +10.05]. No threshold was tuned on held-out outcomes; the full
sweep is reported.

Policy-level tolerance attainment is unchanged by any of these arms (violations per run:
incumbent 1.80, raw-log 1.68, locked 1.68 — vs ideal 0.00): in most cells every affordable
model sits below best−delta, and downrouting neither fixes nor materially worsens that. The
aggregate-quality story must not be oversold.

## 10. Domain-level results (locked arm; 50 cells per domain)

| domain | best-rel opps | inc-rel opps | pool | switch rate | mean dq vs inc | mean dq vs best | valid(best) | C(best) | inferior(inc) |
|---|---|---|---|---|---|---|---|---|---|
| airline | 0 | 40 | 3.0 | 0.10 | −0.002 | −0.042 | 0 | 5 | 1 |
| retail | 0 | 0 | 3.0 | 0.02 | −0.075 | −0.250 | 0 | 1 | 1 |
| telecom | 10 | 10 | 3.0 | 0.14 | +0.425 | 0.000 | 7 | 0 | 0 |

- All best-relative opportunity and all valid switches are **telecom** (gpt-4.1-mini, cheaper
  AND 0.425 better than the incumbent). The 3 misses are pure sample size: exact power 0.540
  at n=3 (§4); outcome variance is secondary (the same pair at n=5 has power ~0.48-0.50 by
  resampling, rising steeply with n because passes need near-all-wins).
- Airline produces the residual best-relative C: gpt-4o-mini is within delta of the incumbent
  (−0.002) but 0.042 below the best — the structural divergence case of §1.
- Retail's single switch (2% rate) is inferior on both definitions — a false switch, counted.
- Aggregate success non-inferiority does not hide a harmful domain: the worst domain-level
  mean dq vs incumbent among switches is retail's −0.075 at a 0.02 switch rate.

## 11. Sensitivity of the locked arm

| setting | A | C | inferior(inc) | save vs incumbent [95% CI] |
|---|---|---|---|---|
| primary (tau=0.95, pool=3-5, delta=0.02) | 0.140 | 0.120 | 0.040 | +5.98 [+1.91, +10.05] |
| tau=0.90 | 0.140 | 0.160 | 0.040 | +6.12 [+2.06, +10.18] |
| tau=0.975 | 0.080 | 0.020 | 0.000 | +2.90 [+0.09, +5.71] |
| pool capped at 2 | 0.067 | 0.073 | 0.033 | +2.72 [+0.19, +5.25] |
| pool capped at 1 | 0.060 | 0.293 | 0.207 | +4.07 [+1.60, +6.53] — savings come with 5x violations; cost CIs alone mislead |
| gate margin 0 / 0.01 / 0.02 / 0.05 | identical to primary (n=3 discreteness, §4) |
| prices x0.75 / x1.25 (non-incumbent) | identical decisions (candidate sets unchanged) |
| paired vs log-anchored | log-anchored: A 0.080, save +5.23 [+1.71, +8.74], audit $0.21 |
| horizon | break-evens are volume-free: ≈ 89 tasks point, ≈ 187 tasks 95%-LB vs incumbent |

The positive lower bound vs the incumbent survives every setting except Bonferroni (§5);
tau = 0.95 was not chosen for being the best cell (tau = 0.90 is slightly better on savings,
worse on C; 0.975 much safer, smaller saving).

## 12. 2x2 ablation (acquisition x anchor), same 50 paired units

| cell | A | C | inferior(inc) | recall(best) | audit $ | cost/1k | save vs inc [CI] | save vs raw-log [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 original acq + best gate (stage 1) | 0.064 | 0.091 | 0.063 | 0.32 | 0.78 | 31.53 | +3.61 [+0.81, +6.41] | +1.40 [−2.75, +5.55] |
| 2 candidate-restricted + best gate | 0.100 | 0.080 | 0.020 | 0.50 | 0.21 | 30.84 | +4.30 [+1.01, +7.59] | +2.09 [−3.46, +7.65] |
| 3 original acq + incumbent gate | 0.060 | 0.158 | 0.101 | 0.30 | 0.78 | 30.73 | +4.42 [+1.15, +7.69] | +2.21 [−1.59, +6.00] |
| 4 candidate-restricted + incumbent gate (locked) | 0.140 | 0.120 | 0.040 | 0.70 | 0.53 | 29.16 | +5.98 [+1.91, +10.05] | +3.77 [−0.94, +8.48] |

**Corrected interpretation (supersedes the earlier analysis file): the two changes are
complementary, and the anchor change alone is NOT sufficient.** Candidate-restricted deep
audits alone (cell 2) raise recall 0.32→0.50 and cut violations; the incumbent anchor alone
(cell 3) increases inferior switches (0.101) because the thin CASE estimates make the
non-inferiority test fire on noise. Only the combination (cell 4) reaches recall 0.70 with
inferior rate 0.040. Success and audit-inclusive cost per cell are in
`free_stage2_validation_summary.csv`.

## 13. Provisional conclusion (conditional on independent validation)

> A targeted, incumbent-anchored paired audit detects substantially more acceptable
> cheaper-model switches than the original best-relative confidence gate and reduces audit
> spending. In the quality-first replay, it has positive lower-bound savings relative to the
> deployed incumbent. Its incremental savings over free raw-log downrouting remain uncertain.

Because the incumbent is the best model in only 40-49% of domain cells, the gate must be
described as incumbent-relative non-inferiority; the wording may be strengthened toward the
cheapest-good-enough objective only where the incumbent is independently shown to be best.

NOT claimable now: that RouterAudit beats free raw-log downrouting; that it is a validated
cost-saving system; that it certifies best-relative quality; that any stage-2 result is
confirmatory; or that the stage-1 no-payback result was erroneous.

## 14. Proposed paper changes (for approval — nothing applied)

**Proposed RQ5 paragraph (replacement for the current final paragraph):**

> The gate's hypothesis, not its evidence, is what binds. Certifying a candidate within
> $\delta$ of the unknown best needs $\sigma^2 z^2/\delta^2 \approx 1{,}700$ audit tasks per
> pair, because a candidate's slack against the best is at most $\delta$; the audit pool
> holds 3-5 tasks per domain. Testing non-inferiority to the deployed incumbent instead needs
> $\sigma^2 z^2/(\text{gap}+\delta)^2$ tasks — about 3 where genuine opportunities exist. In
> a post-hoc replay of this design (candidate-restricted paired audits, incumbent-anchored
> one-sided 95% test), acceptable cheaper-model switches rise from 0.064 to 0.140 per run
> (recall 0.32 to 0.70) at $0.53 audit spend. Relative to the current incumbent the audited
> policy has positive estimated savings with a positive 95% lower bound ($5.98$ per thousand
> tasks, CI $[+1.91,+10.05]$; lower-bound break-even $\approx 187$ tasks). Relative to free
> raw-log downrouting the incremental saving remains uncertain ($+3.77$, CI $[-0.94,+8.48]$).
> Because the incumbent is the true best in under half the domain cells, the gate certifies
> incumbent-relative non-inferiority, not the best-relative tolerance; switches within
> $\delta$ of the incumbent but outside the best-relative tolerance are reported separately.
> These are exploratory replay results pending validation on independent data.

**Proposed Table 1 (columns):** arm x {success, deployment cost/1k, audit $, best-relative
valid A, best-relative violating C, incumbent-relative inferior, saving vs incumbent [95% CI],
saving vs raw-log downrouting [95% CI]}; rows Raw incumbent, Raw-log downrouting, no-audit
log gate (tau=0.9), stage-1 best-relative gate, incumbent-anchored paired audit; footnote:
exploratory, post-hoc; n=3 boundary anti-conservatism 7.6%; both comparators retained.

**Proposed Figure 2:** (a) switches per run for {raw-log, stage-1 gate, incumbent-anchored
audit} stacked by best-relative category with incumbent-relative inferior overlaid as a
marker; (b) paired saving vs incumbent with 95% CIs for the three arms; (c) paired saving vs
raw-log downrouting with 95% CIs. Same style rules as the current figures; all values from
`free_stage2_validation_summary.csv`.

**Claims that would change:** abstract ("identifies only 6% of valid opportunities" → recall
statement + exploratory redesign sentence; final sentence unchanged); contribution 3 (gate
description gains the incumbent-anchored variant, clearly marked exploratory); contribution 4
(unchanged except wording); §5 protocol paragraph (add candidate-restricted paired audit +
non-inferiority test definition + label-cost note); RQ5 (as above); Table 1 and Figure 2 (as
above); Conclusion (add one sentence: hypothesis choice, not evidence volume, bound the
original gate; incumbent-relative savings supported, raw-log comparator still open);
Limitations (add post-hoc/exploratory status and n=3 anti-conservatism); App. C threshold
sweep (add tau sensitivity of the new gate); App. H (new scripts listed). No theorem or
definition changes while the best-relative objective remains primary.
