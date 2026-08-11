# Claim boundaries

What the economic revision licenses, and what it does not. Every sentence added to the paper must
fall on the allowed side of this list.

---

## Allowed, and why

**"On public routing benchmarks, an accurate per-request-type comparison reduces estimated inference
cost at a pre-specified quality tolerance."**
Licensed by Work Package A. A 2-point tolerance saves $1,579 per 1M requests on the primary corpus
and $451 on RouterBench, both against the target-best policy, with paired standard errors over
partitions.

**"A randomized audit of the size this paper studies costs $2.39 ± 0.08 in released benchmark
prices, and repays itself within about 1,500 requests."**
Licensed by A4 and A5. Priced per purchased cell as 30 readouts at the acquisition-cohort cost of the
model in that cell.

**"Deciding from a router-generated log picks a model other than the target-best one in 70.0% of
request types under a 1-point tolerance, against 54.4% deciding from the audited surface and 7.9%
deciding from truth."**
Licensed by A3 combined with the existing pipeline. Replicated on RouterBench.

**"Allocating an audit by dollars rather than by cell count buys 1.8x to 2.5x more evidence for the
same money, and on these corpora that extra evidence did not improve the decision."**
Licensed by A7. The second half is mandatory whenever the first half is stated.

**"Six of eleven models on the released RouterBench roster are strictly dominated on cost and
quality."**
Licensed by direct computation on the released matrix.

## Required companions

Any statement that the audit helps must be accompanied by the finding that **auditing raises
deployment spend** on these corpora, because a confounded log is cheap and wrong. Suppressing it
would make the economic claim misleading.

Any statement of a dollar figure must carry that it is an **estimated counterfactual production cost
computed from released benchmark prices**, not money the project spent and not money a company saved.

Any statement about measurement magnitudes must carry that the distortion is a **pre-specified
synthetic stress test**, not a measured judge.

## Not allowed

- "A real company saved $X." No company data is used.
- "The method increases revenue." Revenue is never observed.
- "The audit reduces inference cost." It does not. It raises it. The tolerance reduces cost.
- "Randomized auditing fixes evaluator bias." Proposition 3.1 says the opposite.
- "The method guarantees safe deployment." The gate bounds harm only on the cells it checks and only
  under the evaluator that checks them.
- "Cost-aware allocation improves decisions." It improves evidence per dollar. Quality was unchanged
  within noise.
- "Released benchmark costs equal actual research spending."
- "LMArena votes represent every user population." Applies once Work Package B runs.
- "The airline benchmark proves airline deployment savings." Applies once Work Package D runs.
- "Assumed annotation prices are observed labour costs."

## Wording substitutions carried into the paper

From the GPT plan's writing checklist, adopted where they do not collide with the paper's existing
notation.

| Instead of | Use, outside the notation table |
|---|---|
| deployment regret | target-value loss, on first use |
| identification error | wrong-model rate, on first use |
| cell | model and request-type pair |
| deconfounded | routing-adjusted, on first use |
| allocation rule | audit-selection strategy, on first use |

The symbols themselves stay as they are, because Table 1, the figures and the generated tables all
depend on them and the notation audit already confirmed every symbol is defined at first mention.
The substitutions apply to the prose gloss at first use only.

## KuaiRand study boundaries (v3, K-family)

Claims the KuaiRand evidence does **not** license:

- "The audit protocol fails in production." One platform, one designated outcome, one arm
  definition; the licensed statement is that the benefit does not separate from zero *here*, and
  why (small selection gap, dominant measurement floor).
- "Real evaluators are biased." The C_kj decomposition is descriptive (one observation per pair,
  residual = interaction) and pool-sensitive: 79.6% (predeclared, scoring pool) vs 87.6% (full
  stream). Both shares travel together or not at all.
- "Usage statistics are generally worthless." Licensed: on this log, traffic share picks a
  target-best arm in no context and anti-correlates with the target (mean Spearman −0.18).
- Any dollar or price statement: the log has no cost column.
- "CASE beats a perfect assignment repair." Weighted-metric-specific (unweighted flips sign,
  n.s.); the licensed statement is that the full click repair never wins and its loss never
  approaches zero.

Mandatory companions wherever K-numbers appear: designated-target framing for `is_like`; the
withheld insertion probability; the absent price column.
