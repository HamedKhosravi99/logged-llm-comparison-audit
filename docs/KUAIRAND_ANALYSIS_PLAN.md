# Analysis plan: external validity on KuaiRand-Pure

Written and frozen **before** any outcome column of KuaiRand was read. Everything above the
"Results" marker at the bottom was fixed while the only quantities inspected were schema, row
counts, cell support counts, and covariate distributions (user counts per context, video counts per
tag, duration percentiles). No `is_click`, `is_like`, or any other outcome column had been loaded
when this plan was frozen.

## Why this study

Every validity claim in the paper rests on offline replays of benchmark matrices with a
*constructed* router and a *synthetic or reconstructed* distortion (App. tests). KuaiRand-Pure
\[gao2022kuairand, CIKM '22, Zenodo 10439422, CC BY 4.0\] is a public production log from a real
platform (Kuaishou) that contains, from the same users over the same window:

- a **selectively assigned stream** — the production recommender chooses video exposure
  (`log_standard_4_22_to_5_08`, 295,497 rows), the real-world A1;
- a **randomized stream** — videos drawn uniformly from a 7,583-item pool and inserted into the
  same feeds (`log_random_4_22_to_5_08`, 1,186,059 rows), a real A0 / audit source;
- a **cheap logged engagement signal** (`is_click`) and **distinct explicit outcomes**
  (`is_like`, `is_hate`) on every row of both streams — a real D axis, not a synthetic one.

So all four A×D validity cells exist in one production dataset with no simulation. The study asks
whether the paper's conclusions — the audit repairs assignment bias, no audit budget repairs
measurement bias, and reading preference off traffic is unreliable — hold on real traffic.

## Mapping to the paper's objects

| paper object | KuaiRand realization |
|---|---|
| candidate model $k$ | content arm: first tag of the video; the **top-8 first-tags by early-window standard-log support** (frozen: 39, 9, 6, 12, 3, 20, 17, 28). Arms are item groups, not the 15 `tab` policies, because the randomized stream randomizes *items*: a uniform draw from the 7,583-item pool lands in every tag group with known positive probability, so per-$(k,j)$ audit pools are well defined, which is false for the serving policies themselves |
| request type $j$ | $J{=}8$ user contexts: `user_active_degree` collapsed to {full, high, middle, other} × follower count {≤100, >100} (frozen before outcomes) |
| router / assignment $A_i$ | the production exposure policies (the standard stream's 15 `tab` policies acting together) |
| logged score $M_{kj}$ | mean evaluator signal in the standard LATE stream (4/22–5/08 only, same window as the random stream, to exclude time confounding) |
| audit readout | mean evaluator signal over $n$ draws with replacement from the random stream's **audit pool** for cell $(k,j)$ (reuses `code/case_core.readout` unchanged) |
| evaluator $Y^o$ (D1) | `is_click` |
| evaluator = target (D0) | `is_like` |
| designated target $\mu^{\mathrm{ref}}_{kj}$ | mean `is_like` over the random stream's **scoring pool** (fixed once, all scoring rows) |
| deployment decision | $\delta{=}0$ per context: pick the best-estimated arm (no price column exists; the economic layer is out of scope here and said so) |

**Roles.** Users are hashed (`sha1(user_id) mod 100`) into three fixed pools: **log pool 40%**
(their standard-late rows form $M$), **audit pool 25%** (their random rows are the readout pools),
**scoring pool 35%** (their random rows define $\mu^{\mathrm{ref}}$ and the click ground truth for
D1 references). The early standard window (4/08–4/21) is the design cohort: it fixed the arm list
above and is never used again. Pools are user-disjoint so no user contributes to more than one of
$M$, readouts, or scoring.

**A0 construction.** As in the paper, A0 holds volume and evaluator fixed and removes selection:
$M^{A0}_{kj}$ is computed from the **log-pool users' random-stream rows**, subsampled per cell to
the same count as that cell's standard-late rows (same users, same window, same evaluator, same
per-cell volume; only the assignment mechanism differs).

**Support rule.** A cell enters $\mathcal K_j$ in a partition when it has ≥10 log-stream rows in
that partition and ≥30 audit-pool rows overall. Global check (already computed): min cell support
is 60 (standard late) and 242 (random), so exclusions should be rare.

## Arms compared (mirroring §5 of the paper)

1. **Trust-the-log** — decisions from raw $M$ (`allocate(mode="raw")`).
2. **Uniform audit** — budget spent on random cells (`mode="uniform"`).
3. **CASE** — closed-loop margin allocation with per-arm offset $b_k$ (`mode="case"`), the paper's
   estimator, code reused verbatim from `code/case_core.py`.
4. **Traffic vote** — choose per context the arm with the largest standard-late exposure count.
   This is "read preference off usage statistics," the instrument an institution reaches for first.
5. **Full-click reference** — decisions from click means on the scoring pool's random rows: the
   ceiling for *any* click-based method (assignment fully repaired, measurement not).
6. **Target reference** — decisions from $\mu^{\mathrm{ref}}$ itself (zero loss by construction).

Each of 1–3 runs under D1 (evaluator `is_click`) and D0 (evaluator `is_like`), giving the four
A×D cells: A1 = standard-stream $M$, A0 = the randomized re-partition above.

## Outcomes

**Primary outcome.** Target-value loss in `is_like` percentage points: for each context, the
scoring-pool like-rate of the best arm minus that of the chosen arm, averaged over supported
contexts (traffic-weighted by context share; unweighted as sensitivity).

**The primary contrast is C1.** C2–C4 are pre-specified secondary contrasts. All are paired over
partitions, at the 35% budget, D1:
- C1: CASE − trust-the-log (does the audit help under real selective assignment?).
- C2: CASE − full-click reference (what residual does the correction leave vs. a perfect
  assignment repair that still scores with click?).
- C3: full-click reference − 0 (the measurement floor: what click can never see).
- C4: traffic-vote − trust-the-log (is the usage heuristic even as good as logged scores?).

**Secondaries.** Wrong-arm rate per context; the same grid under D0 (C2 and C3 should collapse
toward zero — the D-axis check); budget sweep 15/25/35%; Spearman rank correlation, per context,
between standard-late exposure share and scoring-pool like-rate (the "traffic as preference"
diagnostic); decomposition of the evaluator error $C_{kj}=\bar Y^{click}_{kj}-\bar Y^{like}_{kj}$
(scoring pool) into arm main effects + context main effects + residual by two-way ANOVA — the arm
main-effect share is the mass sitting in Cor. blindness form.

## Uncertainty

30 partitions × 3 acquisition seeds. A partition is a hash-split of **log-pool users** into 30
groups; each partition's $M$ uses only that group's standard-late rows, so partitions are
independent replications of a small institution's log at realistic volume (~6.4K rows). Audit
readouts are drawn fresh per partition × seed from the shared audit pool; seeds average within a
partition first; ± is the SE over the 30 partitions; contrasts are paired within partition
(matching `docs/ANALYSIS_PLAN.md` §3). $\mu^{\mathrm{ref}}$ is fixed (full scoring pool), because
the target is a designated constant, not an estimate to be re-randomized — a stated deviation from
the paper's per-partition verification cohorts, forced by `is_like` being a ~2% outcome.

**Readout size.** D1 primary $n_{\mathrm{read}}=30$, the paper's value (`is_click` is a common
event, so 30 draws carry signal). **Declared deviation for D0:** $n_{\mathrm{read}}=300$, because a
30-draw readout of a ~2% binary outcome is mostly zeros and would test nothing but noise; the
random stream makes 300 draws per audited cell realistic for a platform with uniform exploration.
Sensitivity grid for both: $n\in\{30,100,300\}$.

**Seed formula.** `seed = 10_000*partition + 100*int(frac*100) + acq_seed`, matching
`code/eval_validity/pipeline.py`, so contrasts are paired by construction.

## Multiplicity

One primary outcome (target-value loss), one primary budget (35%), one primary evaluator condition
(D1), four predeclared primary contrasts. Everything else is a labelled sensitivity.

## What would count against the paper

- If CASE fails to beat trust-the-log under A1-D1 (C1 ≈ 0 or negative): the audit protocol does
  not transfer to real selective assignment.
- If C3 ≈ 0: real cheap engagement signals rank arms the same as explicit approval, and the
  measurement-validity half of the paper is empirically idle on real traffic.
- If the traffic vote matches the corrected estimators: usage statistics are a fine instrument and
  the OIT framing loses its bite.

Any of these outcomes will be reported as found. **Non-replication rule:** sign reversals and null
results are reported as stated, following the C1 precedent (the LMArena ranking-error non-claim).

## Scope limits, committed in advance

- `is_like` is a platform-intent distinction, not independent welfare; the designated-target
  language of §2 applies verbatim.
- The insertion probability behind the random stream is withheld by Kuaishou, so the marginal
  logging propensity is unpublished; conditional on insertion, exposure over the pool is uniform.
- No cost column exists: none of the paper's economic claims (E-family) extend to this study, and
  no dollar figure will be attributed to it.
- The D-axis here is *measured*, not injected: D0 means the evaluator **is** the designated target
  column, not that a synthetic distortion was removed.
- The propensity-true causal baselines (`ipw_true`, `dr_true`) cannot be run; only
  estimated-propensity variants could be, and are out of scope.

---

## Results marker

Everything below this line was written after outcomes were first computed.

### Declared deviations and verification notes (2026-08-05)

- **ANOVA pool.** The first run computed the $C_{kj}$ decomposition on the full randomized stream;
  the plan predeclares the scoring pool. Caught in adversarial verification (two independent
  paths). Fixed: the code now reports both; the paper quotes the predeclared scoring-pool shares
  (arm 0.796, context 0.091, residual 0.113) as primary and the full-stream shares
  (0.876/0.075/0.050) as a sensitivity. The qualitative claim is unchanged in both.
- **Traffic-vote convention.** 0.892 wrong-arm and 0.00194 loss are means over the 30 partitions'
  realized supports; on the full data the traffic vote picks the most-exposed arm in all 8
  contexts (wrong-arm 8/8, loss 0.0025). The paper labels the convention explicitly.
- **C2 scope.** The CASE-below-perfect-click-repair contrast holds under the traffic-weighted
  primary metric only (unweighted: $+0.00011$ $[-0.00018,+0.00040]$); the paper scopes it.
- **A0 volumes** are approximately matched (~4% of cell-instances under-filled). **D0 readouts**
  are capped at the audit-pool size in 14/64 cells (min 55 rows), a conservative noise direction.
  **Traffic weights and exposure counts** use the full standard-late stream (covariates only; no
  outcome enters). A support guard against NaN target cells was added post-verification; it fires
  nowhere on this dataset.
- All headline inferential numbers (C1--C4, selection gap, Spearman, $C$ grand mean, $\mu^{ref}$)
  passed adversarial verification unchanged: three verifier reports and the adjudication are in
  the session record; independent re-derivations from raw CSVs matched to 4 decimals.
