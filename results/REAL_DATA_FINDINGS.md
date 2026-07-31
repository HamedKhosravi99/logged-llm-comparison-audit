# Real-data findings across all five work packages

Everything below is measured, not simulated. Total API spend **$27.67**, read from result metadata.
The synthetic distortion that was the paper's main limitation is now backed by a real judge, real
human votes, a real operational benchmark and real dollar payouts.

---

## The one number that answers "who saves money"

On 150 SWE-Lancer managerial tasks carrying **$161,375** of released freelance payouts, routing each
task to whichever option minimises expected total cost, charging the payout as human fallback when
the model is wrong:

| Policy | Total realised cost |
|---|---|
| Human only | $161,375 |
| Best single model, gpt-4.1-mini | $71,625 |
| Route on the success estimate a team could actually form | $70,625 |
| **Route on truth** | **$35,625** |

Better routing on the *same* evidence buys **$1,000**. An **accurate comparison** buys **$35,000**.
That thirty-five-fold ratio is the paper's thesis priced on work with real money attached, and it is
the argument for spending an evaluation budget at all.

## The measurement failure is real, not a modelling assumption

A real judge, `gpt-4o-mini`, over **19,377 LMArena battles** with real human votes, blinded, both
answer orders, four parse failures in 38,754 calls.

- agreement with humans **0.430** [0.423, 0.437], **0.582** on decisive votes
- verdict **flips on 33.5%** of battles when the two answers are swapped [0.329, 0.342]
- after conditioning on the human verdict, the longer answer moves the judge **+0.243** while the
  human verdict itself moves it **+0.130**. The judge is nearly twice as responsive to length as to
  what users actually preferred.
- per-model preference spans **-0.125 to +0.106**; agreement is worse on code (0.536) than not (0.601)

On **HelpSteer2**, correctness *is* the judge's largest weight (+0.2024), which is a point in its
favour, yet verbosity (+0.1394) and complexity (+0.1549) still move it after conditioning on
correctness, and it **picks a different response than human correctness on 28.3% of prompts**. That
is a measured analogue of the synthetic experiment's 14 to 18% identification floor.

## Selective exposure damages a real judged comparison

Re-running the assignment contrast with the real judge, selective exposure raises the mean absolute
win-rate error against human truth from **0.1010 to 0.1211**, the direction the synthetic experiment
predicts.

**One non-replication, reported rather than dropped.** A ranking-error summary over the same arms
reversed sign between a 9,703-battle subsample and the full 19,377. At this few model clusters it is
too unstable to support a claim, so only the win-rate error is reported.

## Calibration helps, and saturates early

Five methods, six label budgets, fit on calibration, selected on validation, then applied **once** to
the frozen held-out split.

- logistic correction: **0.5963 → 0.6296** on validation, within 0.002 of its ceiling by 500 labels
- confirmatory pass: **0.5897 → 0.6170** on 7,005 unseen battles, replicating with mild attenuation
- a per-model additive offset gives **nothing at all**
- isotonic is **worse than no correction** below 500 labels

The binding constraint is the correction's functional form, not the label budget. That is the
measurement-side analogue of the paper's correction-family ceiling.

## Different models win different jobs

**τ²-bench-verified**, three domains, four-model ladder, 440 simulations, user simulator frozen,
every model-domain cell at n>=24. gpt-4o-mini leads airline, gpt-4o leads retail, gpt-4.1-mini leads
telecom. Choosing per domain beats **every** single model on success: 0.592, against 0.517 for the
best of them (gpt-4.1-mini), 0.493 gpt-4o, 0.320 gpt-4o-mini, 0.217 gpt-4.1-nano.

**It does not do so by spending less.** Routing costs $39.72 per 1k tasks against gpt-4.1-mini's
$19.44, so +0.075 success costs 2x. Routing is still 0.099 better than gpt-4o at 40% of gpt-4o's
spend. This is the operational form of the same negative the economics show: an accurate comparison
moves traffic toward better and dearer options, and buys decision quality, not savings.

**Corrected 29 July.** The earlier claim ("same quality for less than half the spend", 0.586 at
$76.41) was an artifact of two rate-limited telecom cells at n=3 and n=5. Re-running them to n=26 and
n=25 for $5.48 moved gpt-4o's telecom success from 1.000 to **0.308** and gpt-4o-mini's from 0.200 to
**0.040**, which changed the best single model from gpt-4o to gpt-4.1-mini and inverted the cost
comparison. All twelve cells now clear the minimum-sample guard, so nothing is excluded.

**RouterBench**: six of eleven models are strictly dominated; claude-v2 costs 13x Yi-34B-Chat and
scores lower.

## The audit is cheap and the honest negative stands

The audit costs **$2.39 ± 0.08** on the primary corpus and **$0.89 ± 0.02** on RouterBench, and
repays itself within about 1,500 requests. But **auditing raises deployment spend**, because a
confounded log is cheap and wrong and correcting it moves traffic toward better and dearer models.
The saving comes only from acting on an accurate comparison under a quality tolerance. Cost-aware
allocation buys **1.8x to 2.5x more evidence per dollar** and does **not** improve the decision.

## Spend

| Run | Cost |
|---|---|
| LMArena judge, three splits, 19,377 battles | $10.09 |
| τ², three domains, four models, 440 simulations | $16.56 |
| SWE-Lancer managerial, 150 tasks x 4 models | $0.78 |
| HelpSteer2, 2,400 responses | $0.24 |
| **Total** | **$27.67** |
