# Results: the encouragement design on Criteo-UPLIFT v2.1

First empirical leg of the follow-up paper (see `ENCOURAGEMENT_AND_PREFERENCE_NOTES.md`), run
under the frozen plan `CRITEO_ANALYSIS_PLAN.md` (committed before any outcome column was read;
one dated deviation, the A1 anchor, logged there). Every number below passed an independent
recomputation on a second code path (`code/verify_criteo_encouragement.py`, numpy bincount vs
pandas groupby: all quantities match to <1e-9), the one-sided-compliance check
(P(exposure|control) = 0 exactly), and the per-cell ITT = first-stage x Wald identity.

Data: 13,979,592 users, randomized encouragement (`treatment`, 85.0%), self-selected take-up
(`exposure`, 3.6% of encouraged), cheap proxy (`visit`), designated target (`conversion`).
J=8 covariate cells (median splits of f9, f0, f6 - top full-data variance, outcome-blind);
30 fixed partitions; paired t inference. One cell (w = 12.5%) failed the predeclared
first-stage/arm-size support rule in all partitions and is excluded from pooled summaries.

## The three predeclared contrasts

**K'1 - the log read carries real selection bias, and the instrument removes it by construction.**
The within-encouraged exposed-vs-unexposed contrast (what an operator reading the log would
compute) exceeds the Wald/CACE estimate by **+0.0091 [0.0054, 0.0127]** on conversion (24/30
partitions positive) and **+0.0496 [0.0366, 0.0626]** on visit. Against the Wald conversion
effect of 0.0209 [0.0172, 0.0245], the log read overstates the exposure effect on the target by
roughly 40-70% (per-partition ratio noisy: 0.69 [-0.53, 1.91]; the gap itself is the stable
quantity).

**K'2 - the proxy-target gap largely passes through the instrument.** The visit-minus-conversion
effect difference is **0.196 [0.193, 0.198]** under the naive estimator and **0.155 [0.144,
0.167]** under Wald - **79% of the measurement gap survives instrumenting** (ratio 0.79
[0.74, 0.85], 30/30 partitions). Honest reading against the plan's prediction of "approximately
no shrinkage": the gap shrank 21%, because the proxy-target discrepancy is not purely
arm-constant - Cor. `cor:encourage` protects exactly the constant component, and empirically that
component dominates. The operational sentence: randomizing the encouragement fixed who was
compared and left four fifths of what-the-score-means intact. Deciding on the proxy would still
mislead after a perfect assignment repair.

**K'3 - the selection bias is mostly not cell-constant.** A single global offset fitted on half
the cells explains **15.8% [10.6, 20.9]** of per-cell selection-gap variance across partitions
(25.6% in the one full-data split). The encouragement analogue of the paper's correction-family
result: a one-offset working model would leave most of the structure on the table here too.

## A reportable side-finding

The dataset paper's own headline (Table 1 of arXiv:2111.10106: relative average uplift 68.7%
visit / 37.2% conversion) is **not reproducible from the released file** - ITT/control on the
release gives 27.1% / 59.4% - while the same table's positive-outcome rates (4.70 / 0.29) and
treatment ratio (0.85) reproduce exactly. Consistent with the authors' disclosed non-uniform
privacy subsampling: the released data supports contrasts, not the published marginal-uplift
figures. (This is itself a small instance of the paper's reproducibility theme.)

## Caveats (committed in the plan, plus one learned)

Ad exposure, not LLM model choice - the transfer is structural only. No prices: no dollar
claims. Subsampled rates are not production rates; only contrasts/ratios are reported. Learned in
the run: `exposure -> visit` has a partially mechanical channel in ad delivery (an exposure often
occurs en route to a visit), which inflates proxy-side magnitudes; this strengthens the
proxy-vs-target point but means visit-side effect sizes should not be quoted as behavioral
effects.

## What this buys the follow-up paper

Real-data evidence for both halves of the encouragement corollary: assignment-side repair works
through the rate (K'1), and measurement-side bias survives it (K'2), with the correction-family
limitation reappearing (K'3). Remaining legs per the notes doc: acquisition rules under
instrument-scaled variances, exposure economics, an Oregon-HIE reanalysis as the institutional
parable, and the simulated portal with preference-plus-inertia choice modeling.
