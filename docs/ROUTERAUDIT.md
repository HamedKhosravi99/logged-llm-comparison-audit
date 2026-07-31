# RouterAudit — a practitioner's guide

**RouterAudit** decides whether a routed production log can support a model-comparison decision,
and what evidence to buy next if it cannot. It is the offline tool released with the paper.

It takes a trace export of the shape observability platforms already produce and returns four
things: a **verdict** on whether the log can be trusted for this decision, a **corrected
comparison**, a **purchase list** of the comparisons worth buying, and a **break-even volume** —
the traffic at which the evaluation pays for itself.

It runs entirely offline. It never calls a model API, never changes production routing, and never
claims a saving it cannot support.

---

## The workflow

### 0. Export your traces

One row per served request. Three columns are required:

| column | meaning |
|---|---|
| `request_type` | the segment you want a per-segment decision for (intent, task family, customer tier) |
| `candidate_model` | the model that actually served the request |
| `automatic_score` | whatever score you already log — judge rating, rubric score, thumbs-up |

Five optional columns each unlock more of the output:

| column | what it unlocks |
|---|---|
| `inference_cost_usd` | every dollar figure: deployment cost, savings, break-even |
| `traffic_weight` | weighting request types by expected traffic rather than logged volume |
| `target_score` | the measurement-validity check — the outcome you actually care about |
| `selected_model` | the incumbent policy, so switches are scored against what you run today |
| `request_id` | pairing and de-duplication |

If you run Arize Phoenix or any OpenTelemetry-compatible tracer, `PhoenixAdapter` maps a span
export onto this schema, so step 0 is usually an export rather than an instrumentation project.

### 1. `validate` — can this log answer the question at all?

```bash
python code/router_audit.py validate traces.csv
```

Read the warnings before anything else. They are the point of this step, not noise. It tells you
when there are fewer than two candidate models, so there is no comparison to make; when
model/request-type pairs have too few rows to use; when pairs **never appear in the log at all**,
which is the routed-log hole the audit exists to fill; when costs are missing, which are never
treated as zero; and when there is no `target_score`, so measurement validity cannot be checked.

If this step says the log cannot support the decision, stop. No amount of analysis downstream
fixes a log that never observed the comparison.

### 2. `diagnose` — is the log imbalanced enough to be worth auditing?

```bash
python code/router_audit.py diagnose traces.csv
```

This is the step that decides whether to spend anything. **Auditing an already balanced log makes
the decision worse, not better** — in the paper's experiments it raises target-value loss by
+0.0107. Buy randomized comparisons when the diagnosis says the log is imbalanced; skip them when
it does not.

### 3. `recommend` — what should the next evaluation dollar buy?

```bash
python code/router_audit.py recommend traces.csv --budget-usd 500
```

Returns the purchase list: which model/request-type comparisons to run, in priority order, under
your budget. The tool prioritizes comparisons that can actually change the decision, and each
purchased comparison is 30 readouts, matching the paper's protocol.

### 4. `simulate` — would the switch survive, and when does it pay?

```bash
python code/router_audit.py simulate traces.csv --future-requests 100000 --quality-tolerance 0.01
```

`--quality-tolerance` is the decision you own: how much quality, in your score's own units, you
are willing to give up for a cheaper model. Set it before you look at the output. The tool returns
the recommended policy, its estimated quality and cost, and the break-even volume.

### 5. `report` — the artifact to circulate

```bash
python code/router_audit.py report traces.csv --out report.md
```

A written decision record: the verdict, the corrected comparison, what was bought, what it cost,
what it recommends, and the uncertainty on all of it.

The same five steps are available as a Python API:

```python
from router_audit import AuditStudy
study = AuditStudy.from_csv("traces.csv")
study.validate()
study.diagnose()
plan = study.recommend_audits(budget_usd=500, future_requests=100_000, quality_tolerance=0.01)
print(study.report(plan))
```

---

## How to read what comes back

**A cheaper model is not a saving until three things hold at once.** It must be within your quality
tolerance *on the corrected comparison*, not on the raw log; the saving must survive the cost of
the evidence that established it; and your traffic must be large enough to reach break-even. The
tool reports all three separately so you can see which one fails.

**Auditing raises inference spend before it lowers it.** Correcting a confounded log moves traffic
toward better and dearer models. It is the *quality tolerance* that saves money, not the audit.
Expect the corrected comparison to recommend spending more per request unless you set a tolerance.

**Valid opportunities are rarer than they look.** In the paper's operational replay, a
quality-preserving cheaper alternative existed in only 6.7% of decisions, and 80% of runs
contained none at all. A verdict of "no valid switch here" is the common and correct answer.

## What it does not do

These limits are enforced in the tool, not just documented:

- It does not call model APIs, so it cannot generate new evaluations for you — it tells you which
  ones to buy.
- It does not modify production routing.
- It does not guarantee savings.
- It cannot establish **measurement validity**. Randomizing which model answers which request
  changes who is evaluated on what, not what the score means. If your automatic score disagrees
  with the outcome you actually care about, no audit budget repairs that; supply a `target_score`
  column of validated outcomes instead.
- It never treats a missing cost as zero.

## Where the method comes from

The protocol, its assumptions and its measured limits are in the paper (`tae_workshop_v2.pdf`):
the validity decomposition and the blindness result in §3, the audit protocol in §4, and the
operational evidence in §6. `docs/CLAIM_BOUNDARIES.md` records the wordings the evidence supports
and the ones it does not.
