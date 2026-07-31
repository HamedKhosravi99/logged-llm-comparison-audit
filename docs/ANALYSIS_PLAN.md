# Analysis plan for the economic revision

Covers S1 through S5 and the cost-accounting rules. Written before the confirmatory economic runs
were interpreted; the runs themselves were executed only after `smoke_econ.py` passed every check.

---

## 1. One primary outcome per work package

| Package | Primary outcome | Secondary |
|---|---|---|
| **A**, controlled routing economics | Net savings at 100,000 future requests under a 1-point quality tolerance | target-value loss, wrong-model rate, audit cost, break-even, ROI |
| **B**, LMArena | Wrong user-preference decision rate after routing correction | human-judge agreement, position inconsistency, length bias, model bias, calibration benefit per label |
| **C**, HelpSteer2 | Judge-selection disagreement with the human-correctness choice | dimension weights |
| **D**, tau2 | Cost per successful task at a 2-point non-inferiority tolerance | policy violations, tool errors, turns, latency |
| **E**, SWE-Lancer | Benchmark-implied net savings under model-plus-human-fallback | dollar-weighted success, cost by task-value band |

**Fixed primary settings, chosen before outcomes were compared.** Deployment horizon 100,000
requests. Scalar quality tolerance 0.010. Audit budget the 35% cell fraction the paper already uses.
Evaluator condition A1--D1.

## 2. Multiplicity

One primary outcome, one primary budget, one primary tolerance and one primary horizon per package,
as above. Everything else — the other two horizons, the other three deltas, the four etas, the five
gammas — is reported as a pre-specified sensitivity grid, not as a family of primary tests. The
grids were fixed in `run_econ.py` before the first full run and are recorded in the `spec` block of
every result file.

## 3. Uncertainty

**Controlled corpora.** Per-partition means first, then a paired standard error across partitions,
matching the paper's existing convention. Acquisition seeds are averaged within a partition before
pairing, so seed noise does not inflate the partition count. Savings contrasts are paired against
the same partition's trust arm.

**Where a scheme changes a conclusion, say so.** The paper already reports the two contrasts that
lose separation from zero under slice-cluster resampling. The economic layer inherits that rule.

**LMArena.** Cluster bootstrap by `evaluation_session_id`, never by row.
**HelpSteer2.** Bootstrap by prompt, source model as a random effect.
**tau2.** Bootstrap by task, mixed model over repeated trials.
**SWE-Lancer.** Bootstrap by task, dollar-weighted and unweighted separately.

## 4. Precision checks before final runs

Pilot first, then size. The economics were piloted at 2 and 3 partitions before the 30-partition
run, and the pilot surfaced a real design error — the quality-constrained policy was being computed
only on truth, which cannot answer what an accurate comparison is worth. It was corrected before the
confirmatory run. The 20,000-battle LMArena figure and the four tau2 trials in the plans are design
defaults, not guarantees, and must be re-derived from pilot event rates before the full runs.

## 5. Failure handling, defined in advance

Log every failure. Never silently rerun until success. Count attempts and their cost. Use a fixed
retry policy. Report with and without infrastructure failures where the distinction matters.

**Never set a missing outcome to success. Never set a missing cost to zero.** `econ_core.audit_cost`
raises on a missing acquisition cost rather than treating it as free, and `cost_tables` reports a
per-surface count of missing cells which the smoke test asserts is zero on supported cells.

Failure classes to be logged for the inference packages: API timeout, content-filter refusal,
invalid structured output, tool crash, container failure, judge parse failure, partial trajectory,
missing cost.

## 6. Cost-accounting rules

**Three categories, never merged.** Acquisition-cohort cost prices the audit, verification-cohort
cost prices the gate, scoring-cohort cost prices deployment. `econ_core.cost_tables` returns all
three separately and nothing in the codebase adds them together implicitly.

**Released benchmark cost is not research spend.** Every dollar figure derived here is an estimated
counterfactual production cost computed from costs the corpus authors released. The project did not
pay it. This is stated wherever a figure is reported.

**Break-even is reported as absent, not negative.** Where the per-request saving is nonpositive,
`savings()` returns `None` for `break_even_requests` and the aggregator drops it rather than
reporting a meaningless negative horizon.

**Assumed prices are labelled as assumptions.** The gamma grid for quality valuation and any
per-label annotation price are sensitivities, swept rather than asserted, with no single value
load-bearing.

## 7. Go/no-go gates

| Gate | Condition | Status |
|---|---|---|
| **1. Cost data** | cost fields complete, source documented, no supported cell missing a cost | **passed**, smoke check 2 |
| **2. Cost-aware allocation** | dollar budget enforced, no purchase exceeds the remaining budget, unit tests match hand calculations | **passed**, smoke checks 3 to 7 |
| **3. Reproduction** | committed results reproduce before anything is built on them | **passed**, 0.000e+00 over 129,600 fields |
| **4. LMArena judge** | structured output parses at 99% after fixed retries, both orderings map correctly, names blinded, final-test IDs untouched | not reached |
| **5. tau2** | five-task pilots complete for every candidate model, traces and costs available, failures below threshold | not reached |
| **6. SWE-Lancer** | managerial tasks run, environments stable, failed attempts retain cost logs | not reached |
| **7. Practical claim** | quality constraint met, net savings positive after audit cost, uncertainty not clearly negative, baseline and horizon stated | **conditional**, see below |

**Gate 7 is not met for the claim "the audit reduces cost."** It is met for the claim "an accurate
comparison under a stated quality tolerance reduces cost, and the audit is what makes the comparison
accurate enough to act on." The distinction is enforced in `CLAIM_BOUNDARIES.md` and is the reason
the honest negative in `ECON_FINDINGS.md` section 3 is reported rather than suppressed.
