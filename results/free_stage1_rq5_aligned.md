# Stage 1, aligned to the paper's RQ5 pipeline

No new API calls. Two operational replays of the same 440 $\tau^2$ simulations.
Statistical unit: 20 acquisition seeds averaged inside each fold x split unit, then 50 units.

## Reproduction gate

| quantity | target | tol | reproduced | pass |
|---|---:|---:|---:|:--:|
| raw_success | 0.487 | 0.002 | 0.487 | YES |
| raw_cost_1k | 26.34 | 0.25 | 26.344 | YES |
| case_success | 0.479 | 0.002 | 0.479 | YES |
| case_cost_1k | 28.34 | 0.25 | 28.340 | YES |
| case_audit | 0.83 | 0.05 | 0.831 | YES |
| case_pairs | 7.2 | 0.2 | 7.160 | YES |

**Gate: PASSED** (paper scoring convention: acquisition seed 0). The aligned results below use the 20-seed average that the analysis plan requires, which moves Current CASE from 0.479/\$28.34 to 0.482/\$29.53.

## Quality-repair scenario (paper's fitted-router log)

**Oracle opportunity.** A quality-preserving cheaper alternative exists for **0.0%** of domain decisions; **100%** of runs contain no feasible cost-saving switch at all. Maximum oracle saving \$0.00 per 1k tasks.

> Cost saving is **impossible** in this scenario. Any apparent saving comes from quality-reducing downgrades, and no audit method can be faulted for failing to find savings that do not exist.

| arm | success | \$/1k | audit \$ | \$/success | switches | harmful | benef | harm\|sw | ben\|sw | retained | tol-ok | BE point | BE certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw routed log | 0.4875 | 26.34 | 0.00 | 0.0540 | 0.000 | 0.000 | 0.000 | n/a | n/a | 3.00 | 0.473 | none | none |
| No audit / retain | 0.4875 | 26.34 | 0.00 | 0.0540 | 0.000 | 0.000 | 0.000 | n/a | n/a | 3.00 | 0.473 | none | none |
| Current CASE | 0.4821 | 29.53 | 0.83 | 0.0613 | 1.333 | 0.549 | 0.077 | 0.41 | 0.06 | 1.67 | 0.489 | none | none |
| Inverse-variance CASE | 0.4935 | 33.13 | 0.83 | 0.0672 | 1.119 | 0.368 | 0.085 | 0.33 | 0.08 | 1.88 | 0.537 | none | none |
| Safe CASE | 0.4853 | 26.01 | 0.83 | 0.0536 | 0.050 | 0.050 | 0.000 | 1.00 | 0.00 | 2.95 | 0.469 | 2,469 | 31,044 |
| Safe inverse-variance CASE | 0.4852 | 26.01 | 0.83 | 0.0536 | 0.045 | 0.042 | 0.003 | 0.93 | 0.07 | 2.96 | 0.469 | 2,499 | 51,477 |
| Ideal complete matrix | 0.5875 | 62.27 | 0.00 | 0.1060 | 1.580 | 0.000 | 0.000 | 0.00 | 0.00 | 1.42 | 1.000 | none | none |

Paired 95% intervals over the 50 units (starred rows exclude zero):

| contrast | outcome | mean | 95% CI |
|---|---|---:|---|
| Inverse-variance CASE − Current CASE | success | +0.0114 | [+0.0005, +0.0229] * |
| Inverse-variance CASE − Current CASE | cost/request | +0.0036 | [+0.0015, +0.0058] * |
| Inverse-variance CASE − Current CASE | wrong-good-enough | -0.0480 | [-0.0797, -0.0167] * |
| Inverse-variance CASE − Current CASE | harmful switches | -0.1810 | [-0.2760, -0.0860] * |
| Inverse-variance CASE − Raw routed log | success | +0.0060 | [-0.0142, +0.0279] |
| Inverse-variance CASE − Raw routed log | cost/request | +0.0068 | [+0.0025, +0.0113] * |
| Inverse-variance CASE − Raw routed log | wrong-good-enough | -0.0637 | [-0.1247, -0.0003] * |
| Inverse-variance CASE − Raw routed log | harmful switches | +0.3680 | [+0.2440, +0.5060] * |
| Safe inverse-variance CASE − Raw routed log | success | -0.0023 | [-0.0063, +0.0000] |
| Safe inverse-variance CASE − Raw routed log | cost/request | -0.0003 | [-0.0009, -0.0000] * |
| Safe inverse-variance CASE − Raw routed log | wrong-good-enough | +0.0047 | [-0.0000, +0.0140] |
| Safe inverse-variance CASE − Raw routed log | harmful switches | +0.0420 | [+0.0010, +0.0940] * |
| Safe inverse-variance CASE − No audit / retain | success | -0.0023 | [-0.0062, +0.0000] |
| Safe inverse-variance CASE − No audit / retain | cost/request | -0.0003 | [-0.0009, -0.0000] * |
| Safe inverse-variance CASE − No audit / retain | wrong-good-enough | +0.0047 | [-0.0000, +0.0140] |
| Safe inverse-variance CASE − No audit / retain | harmful switches | +0.0420 | [+0.0010, +0.0950] * |

## Cost-reduction scenario (quality-first incumbent)

**Oracle opportunity.** A quality-preserving cheaper alternative exists for **6.7%** of domain decisions; **80%** of runs contain no feasible cost-saving switch at all. Maximum oracle saving \$38.28 per 1k tasks.

| arm | success | \$/1k | audit \$ | \$/success | switches | harmful | benef | harm\|sw | ben\|sw | retained | tol-ok | BE point | BE certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw incumbent | 0.4850 | 35.14 | 0.00 | 0.0725 | 0.000 | 0.000 | 0.000 | n/a | n/a | 3.00 | 0.400 | none | none |
| No audit / retain | 0.4850 | 35.14 | 0.00 | 0.0725 | 0.000 | 0.000 | 0.000 | n/a | n/a | 3.00 | 0.400 | none | none |
| Raw-log downrouting | 0.4878 | 32.93 | 0.00 | 0.0675 | 0.780 | 0.380 | 0.200 | 0.49 | 0.26 | 2.22 | 0.440 | none | none |
| Current CASE | 0.4718 | 32.81 | 0.78 | 0.0695 | 1.187 | 0.640 | 0.243 | 0.54 | 0.20 | 1.81 | 0.450 | 334 | none |
| Inverse-variance CASE | 0.4797 | 35.13 | 0.78 | 0.0732 | 1.203 | 0.538 | 0.284 | 0.45 | 0.24 | 1.80 | 0.488 | 59,563 | none |
| Safe inverse-variance CASE | 0.4907 | 31.54 | 0.78 | 0.0643 | 0.155 | 0.064 | 0.091 | 0.41 | 0.59 | 2.85 | 0.414 | 216 | 515 |
| Ideal complete matrix | 0.5875 | 62.27 | 0.00 | 0.1060 | 1.800 | 0.000 | 0.200 | 0.00 | 0.11 | 1.20 | 1.000 | none | none |

Paired 95% intervals over the 50 units (starred rows exclude zero):

| contrast | outcome | mean | 95% CI |
|---|---|---:|---|
| Inverse-variance CASE − Current CASE | success | +0.0079 | [+0.0009, +0.0160] * |
| Inverse-variance CASE − Current CASE | cost/request | +0.0023 | [+0.0009, +0.0039] * |
| Inverse-variance CASE − Current CASE | wrong-good-enough | -0.0373 | [-0.0607, -0.0147] * |
| Inverse-variance CASE − Current CASE | harmful switches | -0.1020 | [-0.1670, -0.0370] * |
| Inverse-variance CASE − Raw incumbent | success | -0.0053 | [-0.0233, +0.0128] |
| Inverse-variance CASE − Raw incumbent | cost/request | -0.0000 | [-0.0073, +0.0065] |
| Inverse-variance CASE − Raw incumbent | wrong-good-enough | -0.0877 | [-0.1393, -0.0393] * |
| Inverse-variance CASE − Raw incumbent | harmful switches | +0.5380 | [+0.4260, +0.6590] * |
| Safe inverse-variance CASE − Raw incumbent | success | +0.0057 | [-0.0015, +0.0146] |
| Safe inverse-variance CASE − Raw incumbent | cost/request | -0.0036 | [-0.0067, -0.0013] * |
| Safe inverse-variance CASE − Raw incumbent | wrong-good-enough | -0.0143 | [-0.0283, -0.0023] * |
| Safe inverse-variance CASE − Raw incumbent | harmful switches | +0.0640 | [+0.0230, +0.1160] * |
| Safe inverse-variance CASE − No audit / retain | success | +0.0057 | [-0.0017, +0.0143] |
| Safe inverse-variance CASE − No audit / retain | cost/request | -0.0036 | [-0.0067, -0.0012] * |
| Safe inverse-variance CASE − No audit / retain | wrong-good-enough | -0.0143 | [-0.0287, -0.0030] * |
| Safe inverse-variance CASE − No audit / retain | harmful switches | +0.0640 | [+0.0220, +0.1150] * |


---

## Final classification checks (joint quality + economics)

Switches are classified against the **true best** in the domain, which reconciles the
categories with tolerance satisfaction. The earlier 'beneficial' figure was measured against
the *incumbent* and is withdrawn.

| scenario | arm | sw/run | A valid | B qual-safe | C econ-violating | D invalid | precision | recall | switch rate when no opportunity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quality_repair | Current CASE | 1.333 | 0.000 | 0.384 | 0.600 | 0.349 | 0.00 | n/a | 0.444 |
| quality_repair | Inverse-variance CASE | 1.119 | 0.000 | 0.417 | 0.418 | 0.284 | 0.00 | n/a | 0.373 |
| quality_repair | Confidence-gated CASE | 0.053 | 0.000 | 0.000 | 0.053 | 0.000 | 0.00 | n/a | 0.018 |
| quality_repair | Confidence-gated IV CASE | 0.044 | 0.000 | 0.000 | 0.044 | 0.000 | 0.00 | n/a | 0.015 |
| cost_reduction | Raw-log downrouting | 0.780 | 0.040 | 0.200 | 0.540 | 0.000 | 0.05 | 0.04 | 0.250 |
| cost_reduction | Current CASE | 1.187 | 0.089 | 0.304 | 0.761 | 0.033 | 0.07 | 0.09 | 0.375 |
| cost_reduction | Inverse-variance CASE | 1.203 | 0.089 | 0.381 | 0.700 | 0.033 | 0.07 | 0.09 | 0.381 |
| cost_reduction | Confidence-gated CASE | 0.150 | 0.063 | 0.000 | 0.087 | 0.000 | 0.42 | 0.06 | 0.025 |
| cost_reduction | Confidence-gated IV CASE | 0.155 | 0.064 | 0.000 | 0.091 | 0.000 | 0.41 | 0.06 | 0.026 |

### Against the free alternative (raw-log downrouting), cost-reduction scenario

| outcome | mean | 95% CI | significant |
|---|---:|---|:--:|
| success | +0.0029 | [-0.0106, +0.0170] | no |
| deployment cost/request | -0.0014 | [-0.0054, +0.0025] | no |
| audit-inclusive cost/success | -0.0020 | [-0.0111, +0.0078] | no |
| valid cost-saving switches | +0.0240 | [-0.0090, +0.0630] | no |
| quality-violating economical switches | **-0.4490** | [-0.6400, -0.2680] | **yes** |
| total tolerance-violating switches | **-0.4490** | [-0.6460, -0.2750] | **yes** |

Point break-even 559 tasks; **95%-lower-bound economic break-even: none** (lower bound on the
saving is -0.001986 per request). The earlier 513-task figure was computed against the
do-nothing incumbent, not against the free alternative, and does not survive this comparator.

The 95%-lower-bound economic break-even concerns **aggregate economic cost only**. It is not
a quality guarantee and not a deployment-safety certificate.

### Does the gate find the sparse opportunities?

| | P(switch) | P(valid) | P(violate) |
|---|---:|---:|---:|
| **opportunity exists** — Current/IV CASE | 0.685 | 0.445 | 0.555 |
| **opportunity exists** — Confidence-gated IV | 0.410 | 0.320 | 0.680 |
| **no opportunity** — Current/IV CASE | 0.375/0.381 | 0.000 | 0.549/0.509 |
| **no opportunity** — Confidence-gated IV | **0.026** | 0.000 | 0.579 |

Where an opportunity exists the gate switches **less** often and **less** accurately than
ungated CASE. Its entire benefit is suppressing switches where nothing is available
(0.026 against 0.381). It does not detect the sparse opportunities; it reduces activity.
