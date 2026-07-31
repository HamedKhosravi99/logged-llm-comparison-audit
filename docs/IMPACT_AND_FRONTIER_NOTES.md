# Impact and frontier notes (July 29, 2026)

Working notes for the advisor-requested redirection of the TAE paper toward practical relevance and
real-world impact. Three questions drove the research: what is the frontier of LLM-evaluation
science, how do companies actually evaluate LLMs in production, and who makes or saves money in the
routed-LLM world. Every item below was verified against its source page in July 2026 (arXiv IDs
beginning 25xx are 2025 papers, 26xx are 2026).

The advisor's requirements, translated:

1. **Frontier.** Map the current state of the art on how LLMs are evaluated, in research and practice.
2. **Reality.** Ground the paper in what companies actually do (real tools, real workflows, real software, for example on Hugging Face).
3. **Money.** Show who gets money out of using the method. Results should be convertible to dollars saved or earned.
4. **Usability.** Ship something a team building these systems can actually run, not just analysis.
5. **A case.** Build a concrete, motivating, real-world case first, and let it drive the paper.

The action plan derived from these notes lives at the bottom of this file.

---

## 1. The academic frontier of LLM-evaluation science (2024–2026)

### Evaluation-science and validity position papers

- **Toward an Evaluation Science for Generative AI Systems** (Weidinger, Raji, Wallach, Mitchell, Wang, Salaudeen, Bommasani, Ganguli, Koyejo, Isaac; arXiv, Mar 2025) — manifesto: GenAI evaluation must become a science; metrics must connect to real-world deployment performance. https://arxiv.org/abs/2503.05336 — must-cite.
- **Measuring what Matters: Construct Validity in Large Language Model Benchmarks** (Bean et al.; NeurIPS 2025) — systematic review of 445 LLM benchmarks, 29 expert reviewers; recurring construct-validity failures; eight recommendations. https://arxiv.org/abs/2511.04703 — must-cite.
- **BetterBench** (Reuel et al.; NeurIPS 2024 D&B spotlight) — 25 benchmarks against 46 lifecycle criteria; most fail statistical-significance reporting. https://arxiv.org/abs/2411.12990 — framing.
- **Position: Evaluating Generative AI Systems Is a Social Science Measurement Challenge** (Wallach et al.; ICML 2025) — measurement-theory lenses for GenAI evaluation validity. https://arxiv.org/abs/2502.00561 — framing.
- **Measurement to Meaning: A Validity-Centered Framework for AI Evaluation** (Salaudeen et al.; arXiv, May 2025) — what evaluative claims are licensed by given evidence. https://arxiv.org/abs/2505.10573 — framing.
- **Questionable practices in machine learning** (Leech et al., 2024) — 44 eval-gaming practices. https://arxiv.org/abs/2407.12220 — framing.
- **Benchmark Data Contamination of Large Language Models: A Survey** (Xu et al., 2024). https://arxiv.org/abs/2406.04244 — optional.
- **Institutional agendas** — UK AISI research agenda (May 2025) names advancing AI evaluations a first-class priority; NIST ARIA operationalizes sociotechnical testing of deployed systems. https://www.aisi.gov.uk / https://www.nist.gov/news-events/news/2024/05/nist-launches-aria-new-program-advance-sociotechnical-testing-and — framing.

### Statistics of evals

- **Adding Error Bars to Evals** (Evan Miller, Anthropic; arXiv, Nov 2024) — CLT-based standard errors, paired tests, power analysis for model comparisons. https://arxiv.org/abs/2411.00640 — must-cite.
- **AutoEval Done Right** (Boyeau, Angelopoulos, Yosef, Malik, Jordan; 2024) — prediction-powered inference for model evaluation and ranking. https://arxiv.org/abs/2403.07008 — already cited (boyeau2025autoeval).
- **Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data** (Dorner, Nastl, Hardt; ICLR 2025) — when the judge is no more accurate than the evaluated model, no debiasing can cut required ground-truth labels by more than half. https://arxiv.org/abs/2410.13341 — must-cite.
- **How to Correctly Report LLM-as-a-Judge Evaluations** (Lee et al.; arXiv, Nov 2025, rev. May 2026) — plug-in judge-bias correction with CIs covering test and calibration uncertainty; nearest existing measurement-bias-correction machinery. https://arxiv.org/abs/2511.21140 — must-cite.
- **PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation** (Divekar, Majumder; Jan 2026). https://arxiv.org/abs/2601.18777 — optional.
- **A Statistical Framework for Ranking LLM-Based Chatbots** (Dec 2024) — principled Bradley–Terry/Elo treatment. https://arxiv.org/abs/2412.18407 — optional.

### LLM-as-judge frontier

- **A Survey on LLM-as-a-Judge** (Gu et al.; v5 Mar 2025). https://arxiv.org/abs/2411.15594 — must-cite. Companion: https://arxiv.org/abs/2412.05579.
- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge** (Ye et al.; ICLR 2025) — 12 judge biases, CALM framework; the standard bias catalog. https://arxiv.org/abs/2410.02736 — must-cite.
- **LLM Evaluators Recognize and Favor Their Own Generations** (Panickssery, Bowman, Feng; NeurIPS 2024 oral). https://arxiv.org/abs/2404.13076 — already cited.
- **Prometheus 2** (EMNLP 2024) — flagship open trained judge. https://arxiv.org/abs/2405.01535 — framing.
- **Atla Selene Mini** (Jan 2025) — commercial trained-judge frontier. https://arxiv.org/abs/2501.17195 — optional.
- **JudgeBench** (ICLR 2025) — strong judges barely beat random on hard objective pairs. https://arxiv.org/abs/2410.12784 — framing.
- **RewardBench** (NAACL Findings 2025). https://arxiv.org/abs/2403.13787 — optional.
- **LLMBar** (ICLR 2024) — LLM evaluators misled by superficial qualities. https://arxiv.org/abs/2310.07641 — optional.

### Leaderboard and arena critiques

- **The Leaderboard Illusion** (Singh et al., Cohere Labs et al.; Apr 2025) — undisclosed private testing, selective retraction, unequal sampling distort Chatbot Arena rankings. https://arxiv.org/abs/2504.20879 — must-cite.
- **BenchBench / Benchmark Agreement Testing** (Perlitz, Gera et al., 2024). https://arxiv.org/abs/2407.13696 — framing.
- Arena style-control work: https://arena.ai/blog/style-control/ — optional.

### Closest competitors (evaluation from routed or production logs)

- **Causal methods for LLM development and evaluation** (Frauen, Feuerriegel et al.; KDD 2026, arXiv May 2026) — position/survey; explicitly gives the router-confounding example (harder queries routed to the larger model make naive logged comparisons wrong) but offers no identification theorem, no selection-times-measurement decomposition, no crossed experiment. https://arxiv.org/abs/2605.25998 — competitor / contemporaneity evidence; cite it.
- **Correcting Selection Bias in Sparse User Feedback for LLM Quality Estimation** (Morandi, Viswanathan; May 2026) — production feedback is missing-not-at-random: naive quality 34% vs 83% corrected, selection factors up to 28:1; single-system quality, not model comparison under routing. https://arxiv.org/abs/2605.12177 — competitor; strong motivating stat.
- **Off-Policy Evaluation from Logged Human Feedback** (Bhargava et al.; Jun 2024) — OPE machinery for logged preference feedback; no measurement-bias side. https://arxiv.org/abs/2406.10030 — competitor.
- **Causal LLM Routing: End-to-End Regret Minimization from Observational Data** (May 2025) — treats assignment bias for training the router, not for trusting the evaluation. https://arxiv.org/abs/2505.16037 — competitor.
- **Meta-Router** (Sept 2025) — CATE-based correction of biased preference labels for router training. https://arxiv.org/abs/2509.25535 — competitor (training-time).
- **DARS: From Sampled Outcomes to Capability Distributions** (2026) — single-sample label noise in router supervision. https://arxiv.org/abs/2606.06924 — adjacent (already cited as routing2026supervision? verify).
- **Learning to Route LLMs from Bandit Feedback** (2025) — off-policy router learning from logged partial feedback. https://arxiv.org/abs/2510.07429 — competitor (training-time).

**Gap finding.** No existing paper jointly decomposes router-induced selection bias and judge-induced
measurement bias in logged model comparisons, proves what a randomized audit repairs, and tests it in
a crossed experiment. The three nearest neighbors are all partial. The intersection is open; the
strongest positioning is to cite Frauen et al. as contemporaneous evidence the community sees the
problem, while noting no prior work provides the decomposition, the identification result, or the
crossed test.

### Venues evidencing the conversation

- TAE @ NeurIPS 2026 (target venue): https://tai-eval.github.io/
- EvalEval Coalition (NeurIPS 2024 workshop, continuing as ACL 2026 Workshop on Evaluating Evaluations): https://evalevalai.com/
- NeurIPS 2025 Workshop "Evaluating the Evolving LLM Lifecycle": https://sites.google.com/view/llm-eval-workshop

---

## 2. How companies actually evaluate LLMs in production

### The tooling layer (12 vendors checked)

LangSmith, Braintrust, Arize Phoenix, W&B Weave, Langfuse, Galileo, promptfoo, DeepEval, Ragas,
OpenAI Evals, Vertex AI gen-AI evaluation, AWS Bedrock evaluations. All implement some mix of
LLM-as-judge scoring, human annotation queues, code assertions, and online scoring of production
traces. **None of the twelve documents any handling of selection bias, routed-log bias, confounding,
or statistical significance when comparing models on production traffic; "sampling" appears in their
docs only as a cost knob** (LangSmith). Key data points:

- LangSmith: judge + annotation queues + online evaluators; $39/seat/mo Plus tier; LangChain raised $125M at $1.25B (Oct 2025) with LangSmith the primary revenue driver; Klarna flagship customer. https://docs.langchain.com/langsmith/evaluation
- Braintrust: judge scorers + experiments + online scoring; $36M Series A (a16z), $80M Series B at ~$800M (ICONIQ, Feb 2026); customers: Notion, Stripe, Zapier, Airtable, Instacart, Replit, Cloudflare, Ramp, Dropbox. https://www.braintrust.dev/docs/guides/evals
- Galileo: "Luna" small evaluator models; claims 97% lower eval cost than GPT-3.5 judging, ~$175/month to evaluate 1M queries. https://venturebeat.com/ai/galileos-luna-redefines-genai-evaluation-boasting-97-lower-costs-and-11x-faster-speeds
- Langfuse: MIT-core OSS, ~796k daily PyPI downloads; judge + annotation + user feedback. https://langfuse.com/docs/scores/overview
- AWS Bedrock: human evaluation priced at $0.21 per completed human task. https://aws.amazon.com/bedrock/pricing/
- DeepEval community numbers: GPT-4o-judge evals ≈ $5–10 per 1k items. https://deepeval.com/docs/metrics-introduction
- OpenAI hosted Evals platform being deprecated (read-only Oct 31, 2026). https://developers.openai.com/api/docs/guides/evals

### Survey numbers on practice

- **LangChain State of Agent Engineering** (Nov–Dec 2025, n=1,340): among orgs that evaluate, human review 59.8%, LLM-as-judge 53.3%; offline evals 52.4%, online evals 37.3%; **22.8% of orgs with agents in production run no evaluation at all**; quality is the #1 blocker (~1/3). https://www.langchain.com/state-of-agent-engineering
- **Menlo Ventures 2024**: enterprise genAI spend $13.8B (6x 2023); enterprises typically deploy 3+ foundation models, "routing to different models depending on the use case or results". https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/
- **Menlo Ventures 2025** (n≈500): enterprise genAI spend **$37B** in 2025 (3.2x the restated $11.5B 2024 baseline); foundation-model API spend **$12.5B**; enterprise LLM share Anthropic 40% / OpenAI 27% / Google 21%. https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/
- **a16z CIO survey 2025** (100 CIOs): model selection = external leaderboards ("LM Arena as a Gartner-like filter") + internal golden datasets + developer feedback; 37% use 5+ models, **81% orchestrate 3+ model families**; "for most tasks, all the models perform well enough now — so pricing has become a much more important factor"; switching models "can take a lot of engineering time"; LLM spend expected to grow ~75% next year. https://a16z.com/ai-enterprise-2025/
- **McKinsey State of AI 2025**: 88% of orgs use AI but only 39% attribute any EBIT impact — the adoption-to-value gap. https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
- No major survey publishes evaluation's share of AI budgets — the absence of the line item is itself evidence that evaluation spend is unmanaged.

### Field studies of practitioner behavior

- **Who Validates the Validators?** (Shankar et al., UIST 2024) — "criteria drift": practitioners need criteria to grade outputs but grading outputs changes their criteria; evaluation rests on unstable, subjective ground. https://arxiv.org/abs/2404.12272
- **SPADE** (Shankar et al., VLDB 2024) — production "evals" are typically bolt-on assertions mined from ad-hoc prompt edits; deployed inside LangSmith on 2,000+ real pipelines. https://arxiv.org/abs/2401.03038
- **Building Your Own Product Copilot** (Parnin et al., Microsoft; interviews with 26 engineers) — "every test case was a flaky test"; testing/evaluation the dominant pain point; practice is informal. https://arxiv.org/abs/2312.14231

### Dollar anchors

- API prices (verified July 2026, per 1M tokens in/out): OpenAI GPT-5.6-sol $5/$30 … GPT-5-mini $0.25/$2, GPT-5-nano $0.05/$0.40; Anthropic Claude Fable 5 $10/$50, Opus 4.8 $5/$25, Sonnet 5 $3/$15, Haiku 4.5 $1/$5; Google Gemini 3.x $1.50–$2 in, $7.50–$12 out, 2.5 Flash-Lite $0.10/$0.40. **The spread across tiers is 100–1000x per token, which is what makes routing decisions financially material.** https://developers.openai.com/api/docs/pricing , https://platform.claude.com/docs/en/about-claude/pricing , https://ai.google.dev/gemini-api/docs/pricing
- Eval label costs: AWS human task $0.21/item ($210/1k); LLM-judge ≈$5–10/1k (GPT-4o); Galileo Luna ~$175/month per 1M queries; AlpacaFarm: LLM annotators 50x cheaper than crowdworkers. The 20–1000x human-vs-judge gap is why industry defaulted to LLM-as-judge on convenience samples.
- Cost of wrong or unvalidated quality readings: Klarna's assistant did the work of 700 agents and was estimated at $40M profit improvement (2024), then the strategy was partially walked back in 2025 (over-automation); Air Canada held liable for chatbot's invented policy (Moffatt v. Air Canada, 2024 BCCRT 149); Deloitte Australia refunded part of a A$440k contract over AI-hallucinated citations (Oct 2025).
- Per-outcome pricing exists in the wild: **Intercom Fin charges $0.99 per resolved conversation** — a public, citable value-of-a-correct-answer anchor. https://www.intercom.com/pricing

---

## 3. The routed-LLM ecosystem (who makes money)

### Commercial routers and gateways

- **OpenRouter** — $113M Series B led by CapitalG at **$1.3B valuation (May 2026)**; **~25 trillion tokens/week**, 8M users. Publishes public model rankings and a "State of AI" report computed from **100T tokens of routed request metadata, with no selection-bias correction** — industry-scale model comparison read directly off routed logs. https://openrouter.ai/rankings , https://openrouter.ai/state-of-ai
- **OpenAI GPT-5 real-time router** (Aug 7, 2025) — routing became the default interface for all of ChatGPT; launch-day router failure made GPT-5 "seem way dumber", user revolt forced GPT-4o's return within ~72h; Altman: OpenAI "totally screwed up" the rollout. Largest live routing deployment; its public failure mode was misrouting.
- **AWS Bedrock Intelligent Prompt Routing** — official claim "can reduce costs by up to 30% without compromising on accuracy". https://aws.amazon.com/bedrock/intelligent-prompt-routing
- **Azure AI Foundry model router** (2025); Microsoft engineering post measured 4.5–14.2% savings on test workloads.
- **RouteLLM (LMSYS)** — verified claim: cost reductions "over 85% on MT Bench, 45% on MMLU, 35% on GSM8K" while maintaining 95% of GPT-4 performance. https://www.lmsys.org/blog/2024-07-01-routellm/
- **FrugalGPT** — up to 98% cost reduction matching best single API. **Arcee Conductor** — "up to 99%" per-prompt savings claims. **Martian** — $9M seed + Accenture Ventures; publisher of RouterBench. **Not Diamond**, **Unify** (pivoted), **Requesty** (5% markup), **Portkey** ($15M Series A; claims 24k orgs, $180M+ AI spend managed), **LiteLLM** (55k GitHub stars; cost-based routing strategies), **NVIDIA llm-router blueprint**, **Cloudflare / Kong AI gateways** (semantic routing as standard enterprise features).

### Hugging Face artifacts (the "real software" evidence)

- routellm org (trained routers + data): https://huggingface.co/routellm
- RouterBench dataset (withmartian, 405k outcomes): https://huggingface.co/datasets/withmartian/routerbench
- LLMRouterBench corpus (NPULH; Findings of ACL 2026; 33 models x 21 datasets, 400k+ instances, ~$2.7k API cost to produce, MIT license): https://huggingface.co/datasets/NPULH/LLMRouterBench
- lmarena-ai preference datasets (55k/100k/140k real user battles): https://huggingface.co/lmarena-ai
- Trained judges: prometheus-eval, AtlaAI Selene-1-Mini, Skywork-Critic, BAAI JudgeLM.
- Eval harnesses: EleutherAI lm-evaluation-harness (13.5k stars, backend of the Open LLM Leaderboard), HF lighteval, HF evaluate.
- Community routers: philschmid/modernbert-llm-router + ~25 forks.
- **Gap check: HF searches for "routing evaluation" (models), "eval audit" (Spaces), "log correction" (datasets) return zero relevant artifacts. Nothing on Hugging Face audits or corrects model comparisons made from routed logs. A released audit tool would be first of its kind.**

### Routing papers' evaluation habits

FrugalGPT (2023) through RouterDC, GraphRouter, Router-R1, IRT-Router, CP-Router, DiSRouter (ICLR
2026), Avengers-Pro, RouterEval, and both 2025–2026 routing surveys all evaluate on static,
fully-observed benchmark matrices. **No routing paper audits post-deployment routed logs for model
comparison.** Closest adjacent work is training-time only (Meta-Router, DARS, bandit-feedback
routing). Practitioner content (TrueFoundry) acknowledges a production router "only ever sees the
model it picked" and recommends ad-hoc shadow evaluation — the problem is known, the audit tooling
does not exist.

---

## 4. Action plan (derived; see advisor before executing the paper-facing items)

Priority order, each item tagged with the advisor requirement it serves (R1 frontier, R2 reality,
R3 money, R4 usability, R5 case).

1. **Write the case memo first (R5, R3).** One to two pages: a platform team routes traffic across
   3+ models (the 81% norm), reads "which model for which request type" off its observability
   dashboard, and reallocates traffic and budget on that reading. Wire in the verified numbers: $37B
   enterprise spend, $12.5B on APIs, 100–1000x price spreads, OpenRouter's uncorrected
   rankings-from-logs, the GPT-5 misrouting episode, none-of-12 tools handling selection bias.
   Then state what the paper sells: the audit that says which half of the log's verdict is
   trustworthy, what it costs, and what it can never fix.
2. **Dollarize the existing results (R3).** Both corpora already carry per-request dollar costs
   (LLMRouterBench `C[N,K]`; RouterBench `|total_cost`), currently unused. New script
   (`run_tae_dollars.py`): (a) report each policy's inference cost next to its quality; (b) convert
   regret to dollars per month at stated workload scales, with the value-per-quality-point anchored
   on real per-outcome pricing (Intercom Fin $0.99/resolution) and swept as a sensitivity; (c) audit
   ROI: the ~390 audited requests cost cents-to-dollars in API terms vs the regret removed at scale;
   (d) optionally define the designated target as net value (quality x value − cost) — the paper's
   target-outcome machinery already permits any fixed target, and then regret IS dollars. No new
   inference, days of work.
3. **Measure a real judge (R2; also the #1 reviewer objection).** Check whether the two releases
   include response text; if yes, run one real LLM judge (a cheap API judge and/or an open trained
   judge such as Prometheus-2 / Selene-Mini) on a stratified subsample, compute the measured
   per-cell distortion C_kj against the released gold scores, and rerun the D1 arm with measured
   rather than synthetic distortion. Estimated judging cost is tens of dollars. This converts
   "half the experiment confirms a theorem" into a measured-judge result.
4. **Reframe the paper (R1, R2, R3).** Open with the budget question, not the estimand. Add a
   related-work strand on evaluation in practice (Weidinger; Bean; Miller; Dorner; Lee; Leaderboard
   Illusion; Shankar; Frauen et al. as contemporaneous). Add an "in dollars" readout beside the main
   results and a short "what this means for a team" paragraph per question. Add the stakeholder
   paragraph: who acts on this and what they save.
5. **Ship the artifact (R4).** Wrap `case_core.py` + the Q6 dispersion test into a small package and
   an HF Space: upload a log CSV (request type, model, score, optional cost) → audit-worth verdict,
   corrected comparison with CIs, next-cells-to-buy at budget B, dollar deltas under a price sheet.
   Verified: it would be the first such artifact on Hugging Face.
6. **Real-data path (R2, next paper).** Outreach: OpenRouter (their rankings are our object of
   study), Martian (made RouterBench), eval vendors (LangSmith/Braintrust/Langfuse) as distribution.
   The conclusion's "next paper" (split assignment/measurement audit with gold labels via PPI
   control variates) becomes the eval-budget allocator, demonstrated on a real production log.
