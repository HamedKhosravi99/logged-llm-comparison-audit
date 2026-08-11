# Predeclared analysis plan: the encouragement design on Criteo-UPLIFT v2.1

Status: FROZEN before any outcome column (`visit`, `conversion`) was read. Schema, row-count,
`treatment` share and `exposure` support checks are permitted pre-freeze (design-side columns);
everything else waits until this file is committed. Deviations after freeze are permitted only with
a dated entry in the Deviations log and must be reported in any writeup.

This is the first empirical leg of the follow-up scoped in
`ENCOURAGEMENT_AND_PREFERENCE_NOTES.md`: test, on real randomized-encouragement data, whether
(i) instrumenting removes self-selection bias while (ii) the proxy-target measurement gap passes
through the instrument untouched — the empirical counterpart of Cor. `cor:encourage` in
`tae_workshop_v3.tex`. It does NOT touch tae_workshop_v3, whose claims are frozen.

## Data and provenance

Criteo-UPLIFT v2.1, `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`, CC BY-NC-SA 4.0
(no redistribution: repo ships a download script only). Cite arXiv:2111.10106 (v2.1), not the 2018
AdKDD paper (v1 withdrawn). Expected: 13,979,592 rows; header
`f0..f11,treatment,conversion,visit,exposure`; `treatment` randomized ~85/15 (validated by the
authors' classifier two-sample test; we cite, not re-run); `exposure` is one-sided (control
exposure removed by the authors). Known caveats, committed in advance: rows were sub-sampled
non-uniformly for privacy, so marginal rates are not production rates; features are anonymized
projections; this is ad exposure, not LLM model choice — the mapping is structural, not domain.

## Role mapping (fixed)

| paper object | Criteo column | reading |
|---|---|---|
| encouragement $T_u$ | `treatment` | randomized assignment to the targetable population |
| take-up $D_i$ | `exposure` | the user actually saw the campaign (endogenous) |
| designated target $\Yref$ | `conversion` | the outcome the advertiser values |
| cheap proxy $Y^o$ | `visit` | the engagement signal a log would read |

## Design choices (fixed before outcomes)

- **Contexts.** $J=8$ cells from covariates only: rank `f0..f11` by full-data variance
  (outcome-blind), take the top 3, median-split each. Medians and variances computed once on the
  full file. Cell weights $w_j$ = user share per cell.
- **Partitions.** A fixed permutation of row indices (NumPy `default_rng(777)`), split into
  $30$ equal partitions. All contrasts are computed per partition and paired; $\pm$ is an SE over
  partitions; CIs are paired $t$ intervals.
- **Estimators, per cell $j$ and outcome $Y \in \{$`visit`,`conversion`$\}$:**
  - Naive (the log read): $\hat N_j = \bar Y[D{=}1, T{=}1] - \bar Y[D{=}0, T{=}1]$, computed
    within the encouraged arm so it is a pure self-selection contrast. (Secondary, diagnostic
    only: the all-users version mixing control into the unexposed pool.)
  - ITT: $\bar Y[T{=}1] - \bar Y[T{=}0]$.
  - Wald/CACE: ITT divided by the first stage $\bar D[T{=}1] - \bar D[T{=}0]$; under one-sided
    compliance the denominator is $\approx \Pr(D{=}1 \mid T{=}1)$.
  - **Support rule:** a cell enters pooled summaries only if its partition-level first stage
    $\ge 0.005$ and both `treatment` arms hold $\ge 200$ rows; excluded cells are reported.
- **Derived quantities:** selection gap per outcome $= \hat N_j - \widehat W_j$; measurement gap
  per estimator $= \widehat E_j(\text{visit}) - \widehat E_j(\text{conversion})$; pooled versions
  are $w_j$-weighted.

## Primary contrasts (the K'-family)

- **K'1 (selection bias is real and the instrument removes it):** the pooled selection gap on
  `conversion` is nonzero (CI excludes 0) — the log read is assignment-invalid — and its
  construction guarantees the Wald arm carries none; the reported quantity is the SIZE of the bias
  a log reader absorbs, in units of the Wald effect (ratio with CI).
- **K'2 (measurement bias passes through the instrument):** the pooled measurement gap under
  Wald is nonzero (CI excludes 0) and is NOT smaller than under the naive estimator
  (ratio Wald/naive with CI; the corollary's pass-through predicts $\approx$ no shrinkage —
  instrumenting fixes who was compared, not what the score means).
- **K'3 (is the selection bias cell-constant?):** fit one global offset $\hat b$ = mean selection
  gap on a random half of supported cells (RNG 777), apply to the held-out half; report the share
  of held-out cellwise selection-gap variance explained by the constant (the encouragement
  analogue of the paper's row-constant working-model question).

## Anchors and diagnostics

- **A1:** full-data relative ITT uplifts reproduce the authors' published figures,
  $\approx 68.7\%$ (`visit`) and $\approx 37.2\%$ (`conversion`), tolerance $\pm 2$pp; failure
  halts reporting pending investigation.
- **A2:** one-sided compliance: $\Pr(D{=}1 \mid T{=}0) < 10^{-4}$.
- **A3:** first-stage strength per cell (min, median); covariate balance across $T$ within cells
  (standardized mean differences on `f*`, max reported).
- ITT $=$ first stage $\times$ Wald algebraic identity checked per cell (computational
  correctness).

## Inference and reporting rules

Paired $t$ over $30$ partitions for all primary contrasts; report estimate, CI, and per-partition
sign counts. **Non-replication rule:** if any K' fails (e.g., the measurement gap DOES shrink
under Wald), it is reported as-is with the same prominence — the negative result would bound the
corollary's empirical relevance, which is worth knowing.

## Scope limits (committed)

No price column: no dollar or break-even claims. Single campaign, two arms: no
cheapest-good-enough decision layer, no multi-model claims. `visit`/`conversion` are the
advertiser's signals: "proxy" and "target" are designated roles, not welfare. Ad exposure is not
LLM model choice: findings transfer as structure (encouragement + self-selected take-up +
proxy/target divergence), nothing more. Sub-sampled rates are not production rates; only
contrasts and ratios are reported, never marginal rates as facts about Criteo traffic.

## Verification requirement

Every number quoted in any writeup must pass an independent recomputation on a second code path
(different groupby machinery), plus the A-anchors and the algebraic identity above, before
being cited anywhere.

## Deviations log

- **2026-08-10, A1 anchor replaced.** The planned anchor (relative ITT uplift 68.7% visit /
  37.2% conversion, from Table 1 of arXiv:2111.10106) is **not reproducible from the released
  file**: ITT/control on the release gives 27.1% / 59.4%. The file DOES reproduce Table 1's
  positive-outcome rates (4.70 / 0.29) and treatment ratio (0.85) exactly, and the paper states
  the release was non-uniformly sub-sampled for privacy — the published uplift figures evidently
  describe pre-subsampling data. A1 is therefore replaced by: (i) positive rates match Table 1 to
  2 decimals; (ii) treatment ratio matches 0.85; (iii) arm means reproduce across two independent
  code paths. All three pass. The unreproducibility of the headline uplift from the release is
  itself reportable (and consistent with the release notes' own caveat).
