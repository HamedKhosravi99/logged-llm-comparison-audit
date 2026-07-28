# Claim registry — TAE workshop paper

Every load-bearing claim in `tae_workshop.tex`, mapped to the artifact that supports it, the result
file and script that produced it, the wording the evidence supports, and the stronger wording it does
**not**. Numeric values are keyed into `results/tae_paper_numbers.json`; the key is given so a
reviewer can check any quoted figure without re-running anything.

* Result files: `results/tae_validity.json` (mandatory package, 30 partitions × 6 acquisition seeds),
  `results/tae_validity_appendix.json` (six pre-specified sensitivity sweeps, 20 × 6),
  `results/tae_acquisition_overlap.json` (which cells each rule/surface pair buys, 20 × 6),
  `results/tae_correction_family.json` (correction-family ceiling and richer correctors, 30 × 6),
  `results/tae_intervals.json` (95% CIs under partition / request / slice-cluster resampling),
  `results/tae_replication.json` (the whole design re-run on RouterBench, 11 models, 20 × 6),
  `results/tae_diagnostics.json` (reversal-arithmetic check, assignment-invalidity test, gate
  break-even, equivalence tests),
  `results/tae_paper_numbers.json` (aggregated; every number quoted in the paper).
* Scripts: `code/run_tae_validity.py`, `code/run_tae_appendix.py`, `code/run_tae_overlap.py`,
  `code/run_tae_correction.py`, `code/run_tae_intervals.py`, `code/run_tae_replication.py`,
  `code/run_tae_diagnostics.py`, `code/make_tae_artifacts.py`, package `code/eval_validity/`.
* Regenerate end to end (≈35 min):
  `cd code && python run_tae_validity.py --partitions 30 --acq-seeds 6 && python run_tae_appendix.py && python run_tae_overlap.py && python run_tae_correction.py && python run_tae_intervals.py && python run_tae_replication.py && python run_tae_diagnostics.py && python make_tae_artifacts.py`
* Parent commit of the run reported in the paper: `1a5b036`.
* **Label mapping.** The paper writes the evaluator factor as **D** (distortion): D0/D1/D2. The code
  and every result file key the same factor as **E**: E0/E1/E2 (plus E1k, E2rev in the sweeps). The
  assignment factor A0/A1 is the same in both. Only the display layer of
  `code/make_tae_artifacts.py` renames it.

---

| # | Claim | Artifact | Registry key(s) | Supported wording | Prohibited wording |
|---|---|---|---|---|---|
| C1 | A selective router log names the wrong best model in ~2/3 of contexts; a matched-volume common-population log in ~2/5 | §6 Q1 | `diag.log_misrank_A1` (0.675 ± 0.024), `diag.log_misrank_A0` (0.396 ± 0.026) | "the selective log misranks more often than a matched-volume common-population log, at identical per-cell counts" | "router logs are always wrong"; "the gap is entirely selection bias" (finite-sample noise is in both) |
| C2 | Randomized auditing on a deconfounded surface repairs a substantial *fraction* of the assignment failure | Fig. 1(a), Tab. 4 | `reg.A1.E0.35.{trust,case,oracle_sel}`, `paired.A1.E0.35.case_vs_trust` (−0.0161 ± 0.0028, 27/30), `gap.A1.E0.F_repair_ratio_of_means` (0.32) | "CASE repairs a substantial fraction of router-induced assignment bias — about a third of the gap to an oracle repair — under the stated conditions" | "repairs **most** of the assignment failure"; "CASE removes assignment bias"; "CASE guarantees a correct comparison" |
| C3 | The operative step is deconfounding the acquired surface, not the acquisition rule | Fig. 1, §6 Q2, App. C | `reg.A1.E0.35.{uniform,rawmargin,epig_obs}` (0.0414/0.0402/0.0391) vs `{case,epig_corr}` (0.0338/0.0335); `paired.A1.E0.35.case_vs_epigcorr` (+0.0003 ± 0.0011) | "every observed-surface rule lands together, both deconfounded-surface rules land together below them, and the two deconfounded rules trade places across budgets" | "CASE uniquely dominates active evaluation"; "CASE beats R-EPIG" |
| C19 | Every headline holds on a second released matrix; two secondary findings do not | App. C | `rep.*` keys | "selection, partial repair and the measurement floor all replicate on RouterBench (11 models); the rule tie and the cost of auditing do not, and are reported as corpus-specific" | quoting the rule tie or the auditing cost as general |
| C20 | The 18.3% reversal rate is a property of the stress test, not of any judge | §6 Q4, `code/run_tae_diagnostics.py` | `diagx.D1_*` (predicted 28.8% ± 2.7 vs observed 18.3% ± 2.7 over the same contexts) | "the rate is largely fixed by λ and the margin distribution and carries no information about a deployed judge; the structural claim it illustrates is what survives" | presenting 18.3% as a measurement of real judge bias |
| C21 | Assignment invalidity is testable before committing the budget | §6 Q6, App. C | `diagx.D2_*` (1.06 vs 1.28, AUC 0.79; regret 0.0313 rule / 0.0329 always / 0.0356 never) | "a statistic on the K seed readouts separates A0 from A1 at AUC 0.79 and beats both fixed policies" | quoting the in-sample threshold accuracy as out-of-sample performance |
| C22 | Gating pays only under a stated loss asymmetry | §6 Q5, App. C | `diagx.D3_*` (c* = 0.98 / 1.64 / 2.00 at 15/25/35%) | "a held-out check lowers expected loss only if a harmful switch costs at least c* times a missed improvement" | "the gate reduces harm" without naming the asymmetry |
| C23 | The two ties are equivalence claims, tested as such | App. C | `diagx.D4_*` (MDD 0.0031; TOST p = 1e-4 / 8e-4 at a 0.005 margin; ceiling NOT equivalent, p = 0.11) | "CASE and R-EPIG_corr are equivalent at a pre-specified 0.005 margin; CASE and the family ceiling are not shown equivalent" | calling the CASE / ceiling gap a tie |
| C18 | Two headline effects are weaker than the partition SE suggests | App. D | `ci.*` keys; `results/tae_intervals.json` | "the measurement floor in value and the cost of auditing under A0 both lose separation from zero under a five-slice cluster bootstrap, and are not established across benchmark suites" | quoting either as established beyond this suite; treating the partition SE as generalization uncertainty |
| C16 | The un-repaired part of the assignment gap is the action-level correction family, not the audit budget | §6 Q2c, App. C | `corr.E0.{case,oracle_row}` (0.0338 / 0.0318, paired +0.0020 ± 0.0024), `corr.E0.share_{repaired,left_in_family,outside_family}` (0.28 / 0.05 / 0.66), `corr.E0.bias_residual_not_row_constant` (0.68) | "the best-fitting member of the same family, computed from the truth with no budget, is indistinguishable from CASE at a 35% budget, so more readouts would not help" | "CASE is optimal"; "the correction family is adequate" (66% of the distortion lies outside it) |
| C17 | Qualitative conclusions hold across context granularity; magnitudes do not | App. C | `appx.contexts_J{6,12,16}.*` vs the main J=8 run | "every conclusion holds at J ∈ {6,8,12,16}; the identification floor ranges 10.4–18.3% and J=8 gives the largest of the four" | quoting 18.3% as a constant rather than the top of a range |
| C15 | The tie between the two deconfounded-surface rules is mechanical: on that surface they buy nearly the same cells | §6 Q2, App. C | `overlap.E1.{epig_corrected,margin_observed,epig_observed,uniform}` (0.865 / 0.614 / 0.607 / 0.358), same for `E0` | "swapping the rule while holding the surface changes 13% of the purchases; swapping the surface while holding the rule changes 39%" | "the acquisition rule is irrelevant in general" (measured on this grid, at this budget) |
| C4 | Auditing costs regret where assignment is already common-population | Fig. 2, §6 Q2b | `paired.A0.E0.35.case_vs_trust` (+0.0107 ± 0.0020, 6/30), `paired.A0.E1.35.case_vs_trust` | "a randomized audit should be bought against evidence of assignment invalidity, not by default" | "auditing is harmful"; "never audit" |
| C5 | Assignment correction leaves a measurement floor | Fig. 2, §6 Q3 | `gap.A1.E1.G_sel` (0.0439 ± 0.0029), `gap.A1.E1.G_meas` (0.0030 ± 0.0006), `gap.A1.E1.F_repair` (0.24, mean of partition ratios) and `gap.A1.E1.F_repair_ratio_of_means` (0.30) | "a stable shared evaluator bias remains after assignment correction"; both repair fractions quoted, since they are different estimands | "CASE corrects evaluator bias"; "the floor is negligible"; quoting one repair fraction as if it were the other |
| C6 | The floor is small in value and large in identification | Fig. 2 right | `top1.A1.E1.35.oracle_sel` (0.183 ± 0.027), `diag.reference_ranking_reversal_rate` | "with selection bias removed exactly, the operational evaluator still names the wrong model in 18% of contexts, at a regret cost of 0.0030" | "measurement bias does not matter"; "measurement bias dominates" |
| C7 | Both held-out decision checks cut harmful switches; the same-evaluator check is not worse | Tab. 2 | `gate.E1.35.*.false_switch`, `gate.E1.35.same_vs_ref.false_switch` (−0.011 ± 0.008), `gate.E2.35.same_vs_ref.false_switch` (−0.028 ± 0.012) | "we do not find that same-evaluator verification leaves more harmful switches; it is correlated with the log-derived baseline it checks, so it under-switches" | "a same-evaluator gate is as good as an independent one"; "the gate certifies safe deployment" |
| C8 | The same-evaluator channel's verdict is systematically wrong on some contested contexts | Tab. 2, App. C | `gate.E1.35.same_evaluator.channel_verdict_inversion` (4.6% ± 1.1), `appx.lambda_x1.0.gate.E1.same_evaluator.channel_verdict_inversion` (8.3% ± 1.9), readout sweep | "its population-level verdict disagrees with the target channel's on 4.6% of contested contexts, and that error is flat in the readout size" | "the same-evaluator gate is invalid"; "verification is useless" |
| C9 | Verification earns its cells when the audit is thin, costs value when it is rich | Tab. 2, §6 Q4 | `gate.E1.15.*.reference_regret`, `gate.E1.35.*.reference_regret` | "verification is worth its cells exactly when the audit behind it is too small to trust" | "always gate"; "never gate" |
| C10 | Evaluator-stream mismatch: no detectable effect at the tested magnitude | §6 Q5, App. B | `paired.A1.stream_mismatch` (+0.0010 ± 0.0012), algebra test (r = 0.94, slope 0.46) | "the diagnostic is inconclusive at the tested magnitude; the contamination is algebraically real" | "evaluator comparability does not matter"; "stream mismatch is harmless" |
| C11 | The pre-specified distortion actually reverses target rankings | §4, App. B | `diag.mean_abs_effective_C` (0.0100 ± 0.0007), `diag.clipping_attenuation` (0.476), `diag.median_target_margin` (0.041), `diag.reference_ranking_reversal_rate` (0.183) | "λ is fixed by formula on router-train before any run; clipping attenuates it to 0.48× nominal" | "λ was chosen to produce reversals" |
| C12 | No method reads the designated target outcome before final scoring | App. B | `tests.reference_channel_leakage` in `results/tae_validity.json` | "replacing the scoring outcomes with random values leaves the acquired cells, correction, policy and gate decision identical" | "the protocol is leakage-free in general" (the test covers this pipeline) |
| C13 | Prop. 1's cancellation needs assignment invariance, not merely the same judge — and holds only approximately here | §2 Rem. 1, §4, App. B | `diag.assignment_invariance_residual_A1` (0.0018 ± 0.0001), `..._cohort_floor_A0` (0.0013), `..._mean_abs_selection_bias` (0.0991), `..._residual_over_selection_bias` (0.0185) | "the residual is measured, not assumed: 1.8% contamination of the selection bias the differencing identifies" | "using the same judge on both streams makes the distortion cancel"; "the decomposition is exact" |
| C14 | The measurement result is not an artefact of the capability × difficulty interaction | App. C | `appx.favoritism_Ck.reg.A1.E1k.*`, `appx.favoritism_Ck.top1.A1.E1k.oracle_sel` (0.100 ± 0.021), `appx.favoritism_Ck.gate.E1k.same_evaluator.channel_verdict_inversion` (0.077) | "context-invariant favouritism `C_kj = C_k` cancels from the differencing entirely yet still reverses the target-best model in 10% of contexts after a perfect assignment repair" | "the result holds for any evaluator error structure" (only two forms were tested) |
| C15 | Both the damage and the repair scale with how aggressively the router selects | App. C | `appx.router_eps0.05/0.30/0.50.reg.A1.E0.*`, main run at eps = 0.15 | "the share of trust regret CASE removes falls monotonically from 39% at eps = 0.05 to 10% at eps = 0.50; at eps = 0.05 only 28.6 of 40 cells are supported, so extreme selection also converts bias into missing support" | "auditing is worthwhile at any routing aggressiveness"; "the finding holds for router families other than a cost cascade" |

---

## Claims deliberately *not* made

* No claim that the released benchmark score is the right construct for any deployment — it is a
  **designated target outcome** and the paper says so in the abstract, §2 and §6.
* No claim that a reference verification channel is available in production; it is an experimental
  control and is labelled as a ceiling in §6.
* No claim that the five benchmark slices are five independent studies; intervals are over
  prompt-cohort partitions.
* No claim of unique superiority for the corrected-margin acquisition rule (C3).

## Note on the baseline the gates arbitrate against

The status quo the gates arbitrate against is the **log-derived deployment policy**
(`argmax_k M_kj`), never called the "incumbent" in the paper — the incumbent *router* is a separate,
much weaker baseline reported only in App. C (`appx.baseline_router.*`).
