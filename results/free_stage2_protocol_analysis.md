# Stage 2: why the cost-reduction protocol saves no money, and the core change that fixes it

> **SUPERSEDED in part by `free_stage2_validation.md`.** Two claims below were corrected by
> the validation package: (1) "the anchor change, not the sample allocation, is what binds"
> — the clean 2x2 ablation shows the two changes are complementary and the anchor alone
> INCREASES inferior switches; (2) the word "certified" is retired — use "positive 95%
> lower-bound estimated saving". The stage-1 no-payback result against free raw-log
> downrouting stands and was not an artifact.

No new API calls. Everything below is computed from the stored replay by
`free_stage2_diagnosis.py` and `free_stage2_variants.py`, on the identical 50 fold x split-seed
units, seeds, splits, scoring and costs as stage 1. The stage-1 arm is re-run inside the same
script and must reproduce before any comparison is read (gate PASSED: A 0.064, C 0.091,
total 0.155, audit $0.78).

## 1. Diagnosis: the gate certifies an unwinnable hypothesis

The confidence gate asks P(candidate within delta of the TRUE BEST) >= 0.95. But an
opportunity's candidate is *at most* the best, so its slack against that certificate is at most
delta = 0.02 — and in this replay every true opportunity has slack exactly 0.02 (the candidate
IS the domain best, merely cheaper than the incumbent).

- Sample size needed for the best-anchored certificate: n* = sigma^2 z^2 / slack^2
  ~= 0.25 x (1.645/0.02)^2 ~= **1,691 audit tasks per pair**.
- The audit pool contains **3-5 tasks per domain**.
- Therefore the tau = 0.95 gate can never validly fire; every firing, including the stage-1
  arm's 0.064 valid switches per run, is estimation noise. Its 493-task "certified"
  break-even was luck, not detection.

The deployment decision does not need that hypothesis. The team already runs the incumbent;
the decision-relevant question is non-inferiority to the INCUMBENT with margin delta. True
opportunities here are ~40+ points better than the incumbent in their domain, so the same
formula gives n* ~= **3** — exactly what the pool affords.

## 2. The core protocol change

1. **Candidate-restricted acquisition**: audit only models cheaper than the incumbent, on the
   full per-domain audit pool (instead of spreading 8 margin-prioritized buys over all pairs).
2. **Incumbent-anchored paired non-inferiority gate**: score candidate and incumbent on the
   same audit tasks; switch to the cheapest candidate with
   P(q_cand >= q_inc − delta) >= tau, tau = 0.95 primary (0.80/0.90 reported); SE floored at
   the binomial worst case so tiny tied samples never read as certainty.
   A log-anchored variant uses the incumbent's own log instead of re-scoring it (audit $0.21
   instead of $0.53).

## 3. Results (means over 50 units; savings $/1k tasks, positive = cheaper; paired 95% CIs)

| arm | A | C | recall | audit $ | vs incumbent | 95%-LB break-even | vs raw-log downrouting |
|---|---|---|---|---|---|---|---|
| Raw-log downrouting (free) | 0.040 | 0.540 | 0.20 | 0 | +2.21 [-3.89, +8.31] | none | — |
| S0 Confidence-gated IV CASE (stage 1) | 0.064 | 0.091 | 0.32 | 0.78 | +3.61 [+0.81, +6.41] | 493 tasks | +1.40 [-2.75, +5.55] |
| Best-anchored cert, deep audits (V3) | 0.100 | 0.080 | 0.50 | 0.21 | +4.30 [+1.01, +7.59] | 115 tasks | +2.09 [-3.46, +7.65] |
| **NI gate log-anchored tau=0.95** | 0.080 | 0.140 | 0.40 | 0.21 | **+5.23 [+1.71, +8.74]** | **81 tasks** | +3.02 [-1.78, +7.82] |
| **NI gate paired tau=0.95** | 0.140 | 0.120 | 0.70 | 0.53 | **+5.98 [+1.91, +10.05]** | **187 tasks** | +3.77 [-0.94, +8.48] |
| Oracle (perfect detection) | 0.200 | 0 | 1.00 | 0 | +7.70 [+3.24, +12.17] | — | +5.49 [+0.77, +10.22] |

V3 isolates the mechanism: the same deep candidate-restricted audits under the OLD hypothesis
recover only half the improvement — the anchor change, not the sample allocation, is what binds.

Checks on the winning arm (paired, tau=0.95):
- **Quality holds**: dsuccess vs incumbent +0.0108 [-0.0089, +0.0305] (no significant change,
  point positive). Switch rate where no opportunity exists: 0.043.
- **Residual C switches are benign relative to the deployed system**: only 33% are worse than
  the incumbent beyond delta (mean dq vs incumbent −0.014); raw-log downrouting's C switches
  are 70% worse than the incumbent (mean −0.102, worst −0.325).
- **The three missed opportunities are a pure power limit**: all in telecom with a 3-task
  pool; the candidate is truly 42 points better but an unlucky 3-task draw cannot clear 95%
  (p = 0.52-0.74). No threshold below 0.95 recovers them either; only a larger pool would.

## 4. What can now honestly be claimed, and what still cannot

Supported by paired 95% intervals:
1. Valid cost-saving switches rise 0.064 -> 0.140 per run (recall of true opportunities
   0.32 -> 0.70) while audit spend falls $0.78 -> $0.53 ($0.21 log-anchored).
2. Deployment saving vs the incumbent $5.98/1k tasks, 95% CI [+1.91, +10.05], with a
   95%-lower-bound economic break-even of ~187 tasks (~81 log-anchored) including audit cost.
3. Quality-violating economical switches stay suppressed (0.12 vs the raw log's 0.54), and
   the residual ones are mostly not worse than the deployed incumbent.

Still NOT claimable:
- A certified saving against FREE raw-log downrouting: +3.77 [-0.94, +8.48] crosses zero.
  The oracle ceiling against that comparator is +5.49 [+0.77, +10.22], so the claim is
  achievable in principle at recall ~1.0; the residual gap is the 3-task telecom pool, i.e.
  audit size, not protocol design.

## 5. Honesty notes

- The anchor change was derived from the D1 sample-size analysis BEFORE the variants were
  scored, not selected on outcomes; tau = 0.95 stays the primary threshold (sweep reported).
- One artifact was found and fixed during development: a tied 3-task paired sample floored at
  SE 1e-4 read as certainty; the floor is now the binomial worst case sqrt(0.25/n).
- All of this is one benchmark (440 tau2 simulations), 10 true opportunity cells across 50
  units, driven mainly by the telecom domain. The replay reads stored scores; no new traffic.
- These are replay results, not realized savings for an organization.

## 6. Proposed paper language (NOT yet applied)

Terminology consistent with the agreed vocabulary: "incumbent-anchored non-inferiority gate",
"valid cost-saving switch", "95%-lower-bound economic break-even". The certified-saving
comparator is the deployed incumbent; the raw-log-downrouting comparator remains reported
with its zero-crossing interval and the oracle ceiling alongside.
