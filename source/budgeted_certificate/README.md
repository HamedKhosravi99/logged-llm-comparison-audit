# Prop. 3.4 certificate on a corpus at a stated serving budget

Not part of the submission. Archived here because it answers a reviewer objection the paper does
not currently address, in case a later version wants it.

**The objection.** The title, abstract and theory sell the budgeted program of Eq. (1); every
empirical section runs the quality-constrained rule of Eq. (2). Prop. 3.4, the flagship stopping
rule, is checked only by brute force over 31,724 sampled synthetic instances, never on a corpus at
a stated serving budget. App. A shows the two rules do not reduce to each other, so the existing
empirics do not transfer.

**What this runs.** Eq. (1) on LLMRouterBench (K=5, J=8) at stated budgets, exact MCKP by
enumeration, over 30 partitions. The settledness margin is the largest evaluator-error spread eta
at which Prop. 3.4 still certifies the allocation.

**Findings.**
* A tighter budget makes the allocation more certifiable. Median margin falls from 0.0456 at
  1.25x the cheapest feasible spend to 0.0026 at 8x. Paired within partition, tight minus loose is
  +0.0435 median, positive in 25 of 30 partitions. The 22 tie-free partitions show the same.
* At the paper's own assumed spreads: at eta=0.02, 21 of 30 partitions certify at 1.25x and 0 of 30
  at 8x. The mechanism is that a binding budget prices near-tied rivals out of the feasible set,
  which is a phenomenon rule (2) cannot exhibit.
* The certificate agrees with direct rival-by-rival robust optimality on 44 of 44 checked cases.
* At slack budgets the margin is bounded below by the tightest per-workload quality gap, with
  equality in 7 of 12 slack rows and no violations.

**Caveats.** One corpus; RouterBench at K=11 is 11^8 allocations and needs a solver rather than
enumeration. eta is assumed rather than measured, since evaluator error on this corpus is
constructed. 8 of 30 partitions contain an exact quality tie, which forces a zero margin wherever
the tied workload is selected; margins are reported with and without those partitions.

Entry points: `run30.py` (the 30-partition result), `validate.py` and `validate2.py` (the
brute-force cross-check), `run_budgeted_certificate.py` (single partition, verbose).
