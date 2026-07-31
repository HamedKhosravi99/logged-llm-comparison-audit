# The case. Who spends the money, who loses it, and what this buys them

Two pages, written before the paper edits, to be readable cold. Every external number is verified in
`docs/IMPACT_AND_FRONTIER_NOTES.md`; every internal number is in `results/ECON_FINDINGS.md`
and registered in the result files.

---

## 1. The team

A product serves each request with one of several language models. This is the norm, not a
hypothetical. **81% of enterprises orchestrate three or more model families** and 37% run five or
more. Enterprise generative-AI spend reached **$37B in 2025**, of which **$12.5B went to model
APIs**, growing about 75% a year.

Routing exists because the price spread is enormous. Across the tiers a team actually chooses
between, **per-token prices differ by 100 to 1000x**. On the released RouterBench matrix the spread
in realised cost per request is **72x**, from $46 to $3,293 per million requests, and on the same
prompt GPT-4 costs a median of **51x** what Mistral-7B costs.

That team's observability platform already scores production traces. **53.3% of organisations that
evaluate use an LLM as judge**, 59.8% use human review, and 37.3% run online evaluation on live
traffic. The dashboard shows per-model quality side by side.

## 2. The decision

Someone asks which model should serve coding requests. The dashboard answers for free, from the log.
Traffic moves, and the monthly bill moves with it.

**This is the AI-budget question in the form companies actually meet it.** Not "how much should we
spend on AI" but "given that we are spending it, which model handles which request type, and what
evidence licenses that choice."

## 3. Why the free answer is wrong, twice

The router chose which model saw which request, and it chose on difficulty, so the models sat
different exams. And the judge may not measure the outcome the decision is about, so every model sat
the same exam and the exam was the wrong one. The paper's Proposition 3.1 separates these and states
the exact condition under which the standard repair is valid.

## 4. Why nobody catches it

**Zero of twelve** evaluation platforms checked — LangSmith, Braintrust, Arize Phoenix, W&B Weave,
Langfuse, Galileo, promptfoo, DeepEval, Ragas, OpenAI Evals, Vertex, Bedrock — document any handling
of selection bias, routed-log bias, confounding, or significance when comparing models on production
traffic. "Sampling" appears in their docs only as a cost knob.

**OpenRouter publishes public model rankings computed from roughly 100 trillion tokens of routed
request metadata with no selection-bias correction.** It was valued at $1.3B in May 2026 and moves
about 25 trillion tokens a week. Industry-scale model comparison is being read directly off routed
logs, right now, by the largest gateway in the market.

No routing paper audits post-deployment routed logs for model comparison. The nearest academic
neighbour, Frauen et al. at KDD 2026, names the router-confounding example in a position paper and
offers no decomposition, no identification result and no experiment. **Searches of Hugging Face for
routing evaluation, eval audit and log correction return nothing relevant.** The problem is known and
the tooling does not exist.

## 5. What it costs to be wrong

On the released RouterBench roster, **six of eleven models are strictly dominated** — a cheaper and
more accurate alternative exists. claude-v2 costs **13x** what Yi-34B-Chat costs and scores lower.
A team that picks it from a confounded comparison pays thirteen times the price for negative return.

Measured on our own experiment, applying a one-point quality tolerance to the surface a team actually
holds:

| Decide on | Wrong model, primary | Wrong model, RouterBench |
|---|---|---|
| the log, free | **70.0%** | **56.2%** |
| the audited surface | 54.4% | 44.2% |
| truth | 7.9% | 8.1% |

Downside anchors exist in the wild. Klarna's assistant was credited with $40M of profit improvement
in 2024 and the strategy was partially walked back in 2025. Air Canada was held liable for its
chatbot's invented policy. Deloitte Australia refunded part of an A$440k contract over hallucinated
citations. Vendors sell routing on 30 to 99% cost-reduction claims, all of which are only as good as
the evaluation behind them.

## 6. What this paper sells

Three sentences, each measured.

**The free answer is wrong most of the time.** A router-generated log names a model other than the
target-best one in about two thirds of request types.

**A randomized audit costs one to three dollars and repairs a substantial fraction, and no more.**
Priced exactly: **$2.39 ± 0.08** on the primary corpus, **$0.89 ± 0.02** on RouterBench, for the
readouts the method actually purchases.

**Past a point, more readouts buy nothing, and the remaining error is a measurement floor no audit
budget can touch.** That is the moment to stop buying model reruns and start buying gold labels.

## 7. The budget rule

The advisor's question was: if there is an AI budget, what should it be spent on. The paper contains
the answer and has never stated it as one.

1. **Do not audit a balanced log.** Auditing one *raises* regret by +0.0107. The Q6 dispersion test
   on the first K readouts separates the two regimes at AUC 0.79, so this is checkable in advance
   and the answer is sometimes "keep your money."
2. **Audit where the log is imbalanced, and stop early.** The correction family saturates: the
   no-budget ceiling is within noise of what the method already achieves at a 35% budget, so more
   readouts buy almost nothing.
3. **Spend the remainder on labels, not reruns.** The residual 14 to 18% identification error is a
   measurement floor. Model reruns cannot touch it. Gold labels cost $0.21 per item on AWS human
   tasks or $5 to $10 per thousand from an LLM judge.
4. **Allocate by dollars, not by cells.** Pricing the audit and dividing by it buys **1.8x to 2.5x
   more evidence for the same money.** Report honestly that on these corpora the extra evidence did
   not improve the decision, so the gain is in efficiency, not accuracy.

**Payback.** The audit repays itself in **1,514 requests** on the primary corpus and **1,976** on
RouterBench, against a deployment decision worth hundreds to thousands of dollars per million
requests. Any team above a few thousand requests a month is leaving the decision to chance for the
price of a coffee.

## 8. Who acts on it

**Platform and LLMOps teams** run the audit and the do-not-audit test. **Router and gateway
vendors** — OpenRouter, Martian, Not Diamond, Portkey, Bedrock and Azure model routers — publish or
sell rankings computed exactly this way, so their rankings are the object of study. **Evaluation
vendors** get a shippable feature none of the twelve currently has. And the team the Q6 test tells
*not* to audit keeps its money, which is the cheapest recommendation in the paper.

## 9. What this case does not claim

No real company saved a stated amount. Every dollar figure is computed from released benchmark costs
on public corpora, is labelled as an estimated counterfactual production cost, and is reported as a
break-even quantity or a sensitivity rather than a realised saving. The measurement magnitudes come
from a pre-specified synthetic distortion, not a measured judge. Those boundaries are enforced in
`docs/CLAIM_BOUNDARIES.md`.
