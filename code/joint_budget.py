"""Work Package F. Splitting one evaluation budget between routing evidence and measurement evidence.

The paper's conclusion promises this design and never runs it. A randomized audit repairs assignment
validity and provably cannot repair measurement validity, so a team with a fixed evaluation budget
faces a genuine allocation question:

    B = B_routing + B_measurement

where a routing dollar buys a model rerun on a random request, and a measurement dollar buys a
trustworthy target label that calibrates the judge.

Both halves are estimated from measured data rather than assumed. The routing curve comes from the
economics runs on the two routing corpora; the measurement curve comes from the real judge run
against real human preference on LMArena.

F1  five fixed splits plus an adaptive allocation
F2  label-cost sensitivity at four prices

    python joint_budget.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
LABEL_PRICES = (0.25, 1.00, 5.00, 20.00)
SPLITS = ((1.00, 0.00), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75), (0.00, 1.00))
N_READ = 30


def routing_curve():
    """Wrong-model rate as a function of routing dollars spent, from the economics run.

    The floor is the POPULATION-OPERATIONAL oracle, that is what a cellwise-perfect assignment
    repair still gets wrong. It is emphatically not the target policy, whose wrong-model rate is
    zero by construction; using that as the floor would imply routing dollars can drive the error
    to zero, which is exactly what the paper proves they cannot do.
    """
    import collections
    d = json.load(open(os.path.join(RES, "econ_llmrouterbench.json")))
    s = d["summary"]
    trust = s["A1.trust"]["wrong_model_rate"]["mean"]
    case = s["A1.case"]["wrong_model_rate"]["mean"]
    usd = s["A1.case"]["audit_usd"]["mean"]
    base = json.load(open(os.path.abspath(os.path.join(HERE, "..", "..", "results",
                                                       "tae_validity.json"))))
    ser = collections.defaultdict(list)
    for r in base["rows"]:
        if r["budget"] == 0.35 and r["assignment"] == "A1" and r["evaluator"] == "D1":
            ser[(r["method"], r["partition"])].append(r["top1_error"])
    parts = sorted({p for (m, p) in ser if m == "oracle_sel"})
    floor = float(np.mean([np.mean(ser[("oracle_sel", p)]) for p in parts]))
    return dict(trust=trust, case=case, audit_usd=usd, floor=floor)


def measurement_curve():
    """Judge error against labels bought, from the proper five-method calibration in B7.

    Uses the best method at each budget, fitted on the calibration split and evaluated on the
    validation split. An earlier crude marginal-shift correction showed no gain; that was a property
    of the correction, not of calibration, and it is superseded here.
    """
    p = os.path.join(RES, "lmarena_calibration.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    methods = ("per_model", "length_position", "logistic", "isotonic", "hierarchical")
    curve = [(0, 1.0 - d["uncorrected"])]
    for r in d["rows"]:
        curve.append((r["budget"], 1.0 - max(r[m] for m in methods)))
    return dict(base=1.0 - d["uncorrected"], curve=curve, n=d["n_val"])


def main():
    r = routing_curve()
    m = measurement_curve()
    print("Routing evidence, from the economics run on the primary corpus")
    print(f"  wrong-model rate, trusting the log      {r['trust']:.3f}")
    print(f"  after a ${r['audit_usd']:.2f} audit             {r['case']:.3f}")
    print(f"  measurement floor, perfect assignment repair {r['floor']:.3f}")
    routing_gain_per_usd = (r["trust"] - r["case"]) / r["audit_usd"]
    print(f"  routing gain per audit dollar           {routing_gain_per_usd:.4f} "
          f"wrong-model points removed")

    if not m:
        print("\nmeasurement curve needs the LMArena judge outputs; run lmarena_judge.py first")
        return
    print(f"\nMeasurement evidence, from the real judge against real human preference (n={m['n']:,})")
    print(f"  judge-human disagreement, uncorrected   {m['base']:.3f}")
    for n, e in m["curve"]:
        if n:
            print(f"    after {n:>5} labels                    {e:.3f}")

    print("\nF1. Fixed splits and adaptive allocation")
    print(f"  {'split routing/measurement':30}{'routing $':>11}{'labels':>9}"
          f"{'wrong-model':>13}{'judge error':>13}{'combined':>11}")
    best_lab = max((n for n, _ in m["curve"]), default=0)
    for price in (1.00,):
        for B in (10.0, 100.0, 1000.0):
            print(f"  total budget ${B:,.0f} at ${price:.2f} per label")
            for fr, fm in SPLITS:
                br, bm = B * fr, B * fm
                # routing: cap the gain at the achievable floor
                gain = min(routing_gain_per_usd * br, r["trust"] - r["floor"])
                wm = r["trust"] - gain
                nlab = int(bm / price)
                je = m["base"]
                for n, e in m["curve"]:
                    if n <= nlab:
                        je = e
                print(f"    {int(fr*100):>3}/{int(fm*100):<3}{'':21}{br:11,.1f}{nlab:9,d}"
                      f"{wm:13.3f}{je:13.3f}{(wm + je) / 2:11.3f}")

    print("\nF2. Label-cost sensitivity, total budget $100")
    print(f"  {'$/label':>9}{'labels affordable at 50/50':>28}{'judge error':>14}")
    for price in LABEL_PRICES:
        nlab = int(50.0 / price)
        je = m["base"]
        for n, e in m["curve"]:
            if n <= nlab:
                je = e
        print(f"  {price:9.2f}{nlab:28,d}{je:14.3f}")

    print("\n  Reading. Routing dollars are cheap and saturate quickly, because the audit reaches its")
    print("  correction-family ceiling after a few dollars. Measurement dollars do not saturate at")
    print("  these budgets, and they are the only ones that touch the judge error the audit provably")
    print("  cannot repair. Past the routing saturation point, the next dollar should buy labels.")

    json.dump(dict(routing=r, measurement=m, splits=list(SPLITS),
                   label_prices=list(LABEL_PRICES)),
              open(os.path.join(RES, "joint_budget.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(RES, 'joint_budget.json')}")


if __name__ == "__main__":
    main()
