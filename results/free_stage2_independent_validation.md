# Independent validation of the locked targeted incumbent-anchored paired audit

## Verdict: NO-GO (by the pre-specified mechanical rule)

The locked protocol was frozen in `free_stage2_locked_protocol.yaml` BEFORE this run
(sha256 prefixes — config: `c928a2ede5e79c74`, runner `free_stage2_confirmatory.py`:
`55168ad9888ae976`). No parameter was tuned after viewing results. The corpora
(LLMRouterBench primary, RouterBench replication) appear in the paper's earlier validity and
economic experiments, but no stage-2 threshold or rule was selected using their operational
results; stage 2 was designed on tau2-Bench-Verified only.

| criterion (locked) | LLMRouterBench | RouterBench |
|---|---|---|
| S1 positive 95%-LB audit-inclusive saving vs incumbent (v=$0.25) | **FAIL** (LB −5.29 $/1k) | FAIL (LB −19.35 $/1k) |
| S2 inferior-switch rate ≤ 0.05 | PASS (0.0000) | PASS (0.0000) |
| S3 beats matched-safety free baseline, paired CI excludes zero | **FAIL** (−5.97 [−6.25, −5.69]) | FAIL (−19.35) |
| S4 success non-inferior to incumbent (margin 0.01) | PASS (+0.0001 lower bound) | PASS (0.0000) |

GO required S1–S4 on the primary corpus; S1 and S3 fail on both → **NO-GO: the redesign is
not integrated into the main paper.** Per the pre-registered rule it is retained only as a
limitation / future-work observation, and every tau2 stage-2 result stays labeled a
post-hoc exploratory protocol analysis.

## What actually happened (this is the informative part)

**Error control worked exactly as designed.** Boundary size at the locked n=100:
0.004–0.007 at alpha/k (nominal ≤0.0125/0.005); zero incumbent-relative inferior switches
on either corpus (50 units × 5 audit draws each); verified success exactly preserved. The
gate never made an unsupported switch. S2 and S4 pass everywhere.

**But both corpora contain only near-boundary opportunities.** Design report (printed
before results): opportunity prevalence 0.20 (LLMRouterBench: one domain, mmlupro) and
0.125 (RouterBench: one domain, gsm). The true slack (candidate-minus-incumbent gap + delta)
is +0.039 for mmlupro (gpt-5-chat vs qwen3) and +0.024 for gsm (claude-v2 vs gpt-4).
Calibration at n=100 shows pass probability at those slacks of roughly 0.05–0.17 — the
audit needs n ≈ sigma^2 z^2 / slack^2 ≈ 1,500–4,300 paired tasks to detect them, exactly
the sizing formula from the tau2 diagnosis. The tau2 exploratory result was driven by one
LARGE-slack opportunity (telecom, slack 0.445); no such opportunity exists in either
independent corpus. Consequently the locked audit switched in 6.8% of LLMRouterBench units
(mmlupro only, A=0.044, C=0.024/run) and never on RouterBench.

**Verification cost dominates the economics.** The audit scores ~2,100 (LLMRouterBench) and
~7,700 (RouterBench) outputs per unit. At the pre-specified $0.25/output the total audit
cost is $532 and $1,935 against deployment savings of at most ~$0.06/1k tasks — no
horizon in the verification grid produces a supported payback (at v=$0: lower-bound
break-even 358k tasks on LLMRouterBench; infinite on RouterBench).

**A free baseline is competitive here, unlike in tau2.** With a 30% log partition of a
large corpus, the exploration slice alone gives tens-to-hundreds of logged observations per
cheaper model, so free log-evidence gates CAN act: on LLMRouterBench the free tau=0.95 log
gate saves $0.71/1k at an inferior rate of 0.040. The matched-safety baseline therefore
beats the audit by construction once verification cost is charged (S3). On RouterBench the
log is misleading (all free gates have inferior rates 0.31–0.61), and the matched-safety
baseline is the incumbent itself.

## Domain-level results (locked arm)

| corpus | domain | incumbent | opp | switch rate | mean dq vs inc | A | C | inferior |
|---|---|---|---|---|---|---|---|---|
| LLMRouterBench | mmlupro | qwen3-235b | yes | 0.068 | +0.022 | 0.044 | 0.024 | 0 |
| LLMRouterBench | other 4 | qwen3-235b | no | 0.000 | — | 0 | 0 | 0 |
| RouterBench | gsm | gpt-4-1106 | yes | 0.000 | — | 0 | 0 | 0 |
| RouterBench | other 7 | gpt-4-1106 | no | 0.000 | — | 0 | 0 | 0 |

Full tables: `free_stage2_conf_{llmrb,rb}_domains.csv`. Frontier (cost vs inferior-switch
rate, all arms, both corpora): `fig_stage2_conf_frontier.{pdf,png}` +
`free_stage2_conf_frontier.csv`. Unit-level results, decisions, call-level costs, and
verification grids: `free_stage2_conf_{llmrb,rb}_{units,decisions,costs}.parquet`,
`free_stage2_conf_{llmrb,rb}_{summary,verifgrid}.csv`, `free_stage2_conf_verdict.json`.

## Sensitivity (pre-specified only)

Unadjusted per-candidate alpha (not primary): more switches (A 0.188, inferior rate 0.012 on
LLMRouterBench) but the economics are unchanged (audit-inclusive saving −4.94 $/1k). The
NO-GO does not depend on the multiplicity correction.

## What this means for the paper (no edits made)

Per the locked decision rule, NO paper integration of the redesign as a main result. The
honest disposition, subject to your approval:

1. All existing paper claims stay as they are; nothing in the confirmatory run contradicts
   the paper's current negative-leaning RQ5 conclusions — it REINFORCES them: better
   evaluation is not inherently cost-reducing, and opportunity slack determines what an
   audit of a given size can detect.
2. The tau2 stage-2 redesign remains out of the paper, or appears at most as one
   limitation/future-work sentence, e.g.: "In an exploratory redesign on the same replay, a
   targeted incumbent-anchored paired audit increased valid-switch recall and showed a
   positive lower-bound saving relative to the incumbent; on two independent corpora whose
   opportunities are near the tolerance boundary, a locked confirmatory evaluation made no
   inferior switches but detected no economically supported saving, consistent with the
   audit-sizing bound n ≈ sigma^2 z^2 / slack^2."
3. Not claimable anywhere: validated cost-saving system; certified best-relative quality;
   confirmed payback; beats free raw-log downrouting; stage 1 was erroneous.

The one genuinely portable positive finding across all three corpora is the sizing/anchor
analysis itself (what a given audit budget can and cannot certify), and the demonstrated
policy-level error control (0 inferior switches everywhere at the locked alpha-spending
rule). If you want anything in the paper from stage 2, the defensible candidate is that
methodological point as future work — not an economic claim.
