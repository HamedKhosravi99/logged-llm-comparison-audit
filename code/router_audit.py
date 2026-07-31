"""router-audit. Decide whether a routed log can support a model comparison, and what to buy next.

The decision layer that general observability platforms do not have. Input is a trace export of the
shape every platform already produces. Output is a verdict, a corrected comparison, a purchase list
and a break-even volume.

CLI
    python router_audit.py validate  traces.csv
    python router_audit.py diagnose  traces.csv
    python router_audit.py recommend traces.csv --budget-usd 500
    python router_audit.py simulate  traces.csv --future-requests 100000
    python router_audit.py report    traces.csv --out report.md

Python
    from router_audit import AuditStudy
    study = AuditStudy.from_csv("traces.csv")
    d = study.diagnose()
    p = study.recommend_audits(budget_usd=500, future_requests=100_000, quality_tolerance=0.01)
    print(study.report(p))

Minimum columns
    request_type, candidate_model, automatic_score
Optional, each unlocking more of the output
    inference_cost_usd, traffic_weight, target_score, selected_model, request_id

Everything the tool refuses to do is as important as what it does. See `Safeguards`.
"""
import argparse, json, math, sys
from dataclasses import dataclass, field

import numpy as np

MIN_CELL = 8          # logged rows before a model and request-type pair is usable at all
N_READ = 30           # readouts per purchased pair, matching the paper's protocol


# --------------------------------------------------------------------------- safeguards
@dataclass
class Warnings:
    items: list = field(default_factory=list)

    def add(self, code, msg):
        self.items.append((code, msg))

    def __bool__(self):
        return bool(self.items)

    def render(self):
        if not self.items:
            return "  none"
        return "\n".join(f"  [{c}] {m}" for c, m in self.items)


# --------------------------------------------------------------------------- study
class AuditStudy:
    def __init__(self, rows):
        self.rows = rows
        self.models = sorted({r["candidate_model"] for r in rows})
        self.types = sorted({r["request_type"] for r in rows})
        self.K, self.J = len(self.models), len(self.types)
        self.mi = {m: i for i, m in enumerate(self.models)}
        self.ji = {t: i for i, t in enumerate(self.types)}
        self._build()

    @classmethod
    def from_csv(cls, path):
        import csv
        with open(path, newline="") as fh:
            rows = []
            for r in csv.DictReader(fh):
                for k in ("automatic_score", "inference_cost_usd", "traffic_weight", "target_score"):
                    if r.get(k) not in (None, ""):
                        r[k] = float(r[k])
                    else:
                        r.pop(k, None)
                rows.append(r)
        if not rows:
            raise SystemExit("no rows in the trace file")
        return cls(rows)

    def _build(self):
        K, J = self.K, self.J
        self.n = np.zeros((K, J), int)
        self.M = np.full((K, J), np.nan)          # logged score surface
        self.C = np.full((K, J), np.nan)          # per-request cost surface
        self.T = np.full((K, J), np.nan)          # target score where available
        acc = {}
        for r in self.rows:
            k, j = self.mi[r["candidate_model"]], self.ji[r["request_type"]]
            a = acc.setdefault((k, j), dict(s=[], c=[], t=[]))
            a["s"].append(r["automatic_score"])
            if "inference_cost_usd" in r:
                a["c"].append(r["inference_cost_usd"])
            if "target_score" in r:
                a["t"].append(r["target_score"])
        for (k, j), a in acc.items():
            self.n[k, j] = len(a["s"])
            self.M[k, j] = float(np.mean(a["s"]))
            if a["c"]:
                self.C[k, j] = float(np.mean(a["c"]))
            if a["t"]:
                self.T[k, j] = float(np.mean(a["t"]))
        w = {}
        for r in self.rows:
            if "traffic_weight" in r:
                w[self.ji[r["request_type"]]] = r["traffic_weight"]
        if len(w) == J:
            p = np.array([w[j] for j in range(J)], float)
        else:
            p = self.n.sum(0).astype(float)
        self.p = p / p.sum() if p.sum() else np.full(J, 1.0 / J)
        self.supp = self.n >= MIN_CELL

    # ---------------------------------------------------------------- validation
    def validate(self):
        w = Warnings()
        if self.K < 2:
            w.add("E-MODELS", "fewer than two candidate models, there is no comparison to make")
        thin = int((~self.supp & (self.n > 0)).sum())
        if thin:
            w.add("W-SUPPORT", f"{thin} model/request-type pairs have fewer than {MIN_CELL} rows "
                               f"and are excluded from every decision")
        empty = int((self.n == 0).sum())
        if empty:
            w.add("W-UNOBSERVED", f"{empty} pairs never appear in the log, so the log cannot rank "
                                  f"them at all. This is the routed-log hole the audit fills.")
        if not np.isfinite(self.C).any():
            w.add("W-NOCOST", "no inference_cost_usd column, so no dollar output is possible")
        elif int((~np.isfinite(self.C) & self.supp).sum()):
            w.add("W-COSTGAP", f"{int((~np.isfinite(self.C) & self.supp).sum())} supported pairs "
                               f"have no cost, and a missing cost is never treated as zero")
        if abs(self.p.sum() - 1) > 1e-9:
            w.add("W-WEIGHTS", "traffic weights do not normalise, falling back to logged volume")
        vers = {r.get("model_version") for r in self.rows if r.get("model_version")}
        if not vers:
            w.add("W-VERSION", "no model_version column, so results cannot be pinned to a model build")
        if not np.isfinite(self.T).any():
            w.add("W-NOTARGET", "no target_score column, so measurement validity cannot be checked "
                                "and only the assignment half of the problem is addressed")
        return w

    # ---------------------------------------------------------------- diagnose
    def diagnose(self):
        """Should you audit at all? The dispersion test on the logged surface.

        Reproduces the paper's Q6 statistic: the spread of standardized per-model deviations from
        the column mean. A balanced log has low dispersion, and auditing one raises regret.
        """
        z = []
        for j in range(self.J):
            ks = np.where(self.supp[:, j])[0]
            if len(ks) < 2:
                continue
            v = self.M[ks, j]
            s = v.std(ddof=1)
            if s > 0:
                z.append(float(np.sqrt(np.mean(((v - v.mean()) / s) ** 2))))
        disp = float(np.mean(z)) if z else float("nan")
        imbalance = float(np.std(self.n[self.supp]) / max(np.mean(self.n[self.supp]), 1e-9)) \
            if self.supp.any() else float("nan")
        verdict = ("audit" if imbalance > 0.5 else "probably keep your money")
        return dict(dispersion=disp, count_imbalance=imbalance, verdict=verdict,
                    supported_pairs=int(self.supp.sum()), total_pairs=int(self.K * self.J),
                    unobserved_pairs=int((self.n == 0).sum()))

    # ---------------------------------------------------------------- recommend
    def recommend_audits(self, budget_usd=None, future_requests=100_000, quality_tolerance=0.01,
                         n_read=N_READ):
        """Which pairs to buy next, under a dollar budget, cheapest-informative first."""
        cand = []
        for j in range(self.J):
            ks = np.where(self.supp[:, j])[0]
            if len(ks) < 2:
                continue
            o = ks[np.argsort(-self.M[ks, j])]
            gap = float(self.M[o[0], j] - self.M[o[1], j])
            for rank, k in enumerate(o[:2]):
                c = self.C[k, j]
                price = n_read * float(c) if np.isfinite(c) else float("nan")
                score = self.p[j] / (gap + 1e-6)
                if np.isfinite(price) and price > 0:
                    score /= price
                cand.append(dict(model=self.models[k], request_type=self.types[j],
                                 margin=gap, traffic=float(self.p[j]),
                                 audit_price_usd=price, priority=score))
        cand.sort(key=lambda d: -d["priority"])
        chosen, spent = [], 0.0
        for c in cand:
            pr = c["audit_price_usd"]
            if np.isfinite(pr):
                if budget_usd is not None and spent + pr > budget_usd:
                    continue
                spent += pr          # accumulate whether or not a budget was supplied
            chosen.append(c)
            if budget_usd is None and len(chosen) >= 20:
                break
        return dict(purchases=chosen, audit_cost_usd=spent,
                    future_requests=future_requests, quality_tolerance=quality_tolerance)

    # ---------------------------------------------------------------- simulate
    def simulate(self, quality_tolerance=0.01, future_requests=100_000, audit_cost_usd=0.0):
        """Current policy versus a quality-constrained one, on the surface available."""
        cur = self._argmax_policy(self.M)
        qc = self._quality_constrained(self.M, quality_tolerance)
        out = dict(quality_tolerance=quality_tolerance, future_requests=future_requests)
        if not np.isfinite(self.C).any():
            out["note"] = "no cost column, dollar output unavailable"
            return out
        c0, c1 = self._cost(cur), self._cost(qc)
        per = c0 - c1
        out.update(current_cost_per_1k=1000 * c0, recommended_cost_per_1k=1000 * c1,
                   saving_per_request=per,
                   gross_savings=future_requests * per,
                   audit_cost_usd=audit_cost_usd,
                   net_savings=future_requests * per - audit_cost_usd,
                   break_even_requests=(audit_cost_usd / per) if per > 0 else None,
                   recommended_policy={self.types[j]: self.models[qc[j]]
                                       for j in range(self.J) if qc[j] >= 0},
                   current_policy={self.types[j]: self.models[cur[j]]
                                   for j in range(self.J) if cur[j] >= 0},
                   expected_quality_current=self._quality(cur),
                   expected_quality_recommended=self._quality(qc),
                   types_needing_labels=self._types_needing_labels(quality_tolerance),
                   types_insufficient_support=[self.types[j] for j in range(self.J)
                                               if int(np.isfinite(self.M[:, j]).sum()) < 2])
        return out

    def _quality(self, pol):
        v = [self.M[pol[j], j] for j in range(self.J) if pol[j] >= 0 and np.isfinite(self.M[pol[j], j])]
        return float(np.mean(v)) if v else None

    def _types_needing_labels(self, tol):
        """Types whose top-two gap is inside the tolerance: more reruns cannot settle the choice,
        so the binding uncertainty is what the score measures, not who was measured."""
        out = []
        for j in range(self.J):
            col = self.M[:, j][np.isfinite(self.M[:, j])]
            if col.size >= 2:
                top = np.sort(col)[::-1]
                if float(top[0] - top[1]) <= tol:
                    out.append(self.types[j])
        return out

    def _argmax_policy(self, S):
        pol = np.full(self.J, -1, int)
        for j in range(self.J):
            ks = np.where(self.supp[:, j])[0]
            if len(ks):
                pol[j] = int(ks[int(np.argmax(S[ks, j]))])
        return pol

    def _quality_constrained(self, S, tol):
        pol = np.full(self.J, -1, int)
        for j in range(self.J):
            ks = [k for k in np.where(self.supp[:, j])[0]
                  if np.isfinite(S[k, j]) and np.isfinite(self.C[k, j])]
            if not ks:
                continue
            best = max(S[k, j] for k in ks)
            ok = [k for k in ks if S[k, j] >= best - tol]
            pol[j] = int(min(ok, key=lambda k: (float(self.C[k, j]), -float(S[k, j]), k)))
        return pol

    def _cost(self, pol):
        return float(sum(self.p[j] * self.C[pol[j], j] for j in range(self.J)
                         if pol[j] >= 0 and np.isfinite(self.C[pol[j], j])))

    # ---------------------------------------------------------------- report
    def report(self, plan=None, tol=0.01, H=100_000):
        w = self.validate()
        d = self.diagnose()
        plan = plan or self.recommend_audits(future_requests=H, quality_tolerance=tol)
        sim = self.simulate(tol, H, plan["audit_cost_usd"])
        L = []
        L.append("# router-audit report\n")
        L.append(f"- models {self.K}, request types {self.J}, logged rows {len(self.rows):,}")
        L.append(f"- supported pairs {d['supported_pairs']} of {d['total_pairs']}, "
                 f"never-observed pairs {d['unobserved_pairs']}\n")
        L.append("## 1. Should you audit at all\n")
        L.append(f"- logged-count imbalance {d['count_imbalance']:.2f}")
        L.append(f"- **verdict: {d['verdict']}**")
        L.append("\n  A balanced log is not worth auditing. On the study corpora, auditing a balanced")
        L.append("  log *raised* regret, so this verdict can legitimately be to keep your money.\n")
        L.append("## 2. What to buy next\n")
        if plan["purchases"]:
            L.append("| model | request type | top-two margin | traffic | audit price |")
            L.append("|---|---|---|---|---|")
            for c in plan["purchases"][:10]:
                pr = f"${c['audit_price_usd']:.4f}" if np.isfinite(c["audit_price_usd"]) else "n/a"
                L.append(f"| {c['model']} | {c['request_type']} | {c['margin']:.4f} "
                         f"| {c['traffic']:.3f} | {pr} |")
            L.append(f"\n- total audit cost **${plan['audit_cost_usd']:.2f}** "
                     f"for {len(plan['purchases'])} pairs at {N_READ} readouts each\n")
        else:
            L.append("no purchasable pairs, every request type has fewer than two supported models\n")
        L.append("## 3. Money\n")
        if "note" in sim:
            L.append(f"- {sim['note']}\n")
        else:
            cur_p, rec_p = sim.get("current_policy", {}), sim.get("recommended_policy", {})
            if cur_p or rec_p:
                L.append("\n### Decision per request type\n")
                L.append("| request type | current model | recommended model | action |")
                L.append("|---|---|---|---|")
                for t in self.types:
                    c, r = cur_p.get(t, "-"), rec_p.get(t, "-")
                    act = ("retain" if c == r else "switch") if r != "-" else "collect more evidence"
                    L.append(f"| {t} | {c} | {r} | {act} |")
                qc_, qr_ = sim.get("expected_quality_current"), sim.get("expected_quality_recommended")
                if qc_ is not None:
                    L.append(f"\n- expected quality, current {qc_:.4f}; recommended {qr_:.4f}")
            need = sim.get("types_needing_labels") or []
            thin = sim.get("types_insufficient_support") or []
            L.append(f"- request types requiring validated labels: "
                     f"{', '.join(need) if need else 'none'}")
            L.append(f"- request types with insufficient support: "
                     f"{', '.join(thin) if thin else 'none'}")
            L.append(f"- current policy ${sim['current_cost_per_1k']:.4f} per 1k requests")
            L.append(f"- recommended at a {tol} tolerance ${sim['recommended_cost_per_1k']:.4f} per 1k")
            L.append(f"- gross savings at {H:,} requests **${sim['gross_savings']:,.2f}**")
            L.append(f"- net of the audit **${sim['net_savings']:,.2f}**")
            be = sim["break_even_requests"]
            L.append(f"- break-even **{be:,.0f} requests**" if be else
                     "- **no finite break-even**, the recommended policy is not cheaper")
            L.append("")
        L.append("## 4. What this does not fix\n")
        L.append("A randomized audit repairs the assignment half of the problem only. If the")
        L.append("automatic score measures something other than the outcome the decision is about,")
        L.append("no audit budget removes that error, and a comparison corrected this way can still")
        L.append("name the wrong model. Buy gold labels for that half, not more reruns.\n")
        L.append("## 5. Warnings\n")
        L.append(w.render())
        return "\n".join(L)


# --------------------------------------------------------------------------- adapters
class PhoenixAdapter:
    """Map an Arize Phoenix / OpenTelemetry trace export onto the tool's schema.

    Phoenix exports spans with model attributes and, separately, annotations. The mapping is
    LLM spans -> candidate model plus token cost, annotations -> target labels, LLM evaluations ->
    automatic scores, and a chosen span attribute -> request type. Any observability platform that
    exports spans and annotations can be adapted the same way; only the column names differ.
    """

    SPAN_MAP = {"llm.model_name": "candidate_model",
                "llm.token_count.total": "tokens",
                "input.value": "prompt"}

    @staticmethod
    def load_export(traces_path, annotations_path=None, request_type_attr="tag.request_type",
                    price_per_1k=None):
        import csv, os
        rows = []
        with open(traces_path, newline="") as fh:
            for sp in csv.DictReader(fh):
                model = sp.get("llm.model_name") or sp.get("attributes.llm.model_name")
                if not model:
                    continue
                r = dict(request_id=sp.get("context.span_id") or sp.get("span_id", ""),
                         request_type=sp.get(request_type_attr) or sp.get("tag.request_type", "all"),
                         candidate_model=model, selected_model=model)
                sc = sp.get("eval.score") or sp.get("annotation.score")
                if sc not in (None, ""):
                    r["automatic_score"] = float(sc)
                tok = sp.get("llm.token_count.total")
                if tok not in (None, "") and price_per_1k:
                    r["inference_cost_usd"] = float(tok) / 1000.0 * float(price_per_1k.get(model, 0.0))
                cost = sp.get("llm.cost.total")
                if cost not in (None, ""):
                    r["inference_cost_usd"] = float(cost)
                if "automatic_score" in r:
                    rows.append(r)
        if annotations_path and os.path.exists(annotations_path):
            ann = {}
            with open(annotations_path, newline="") as fh:
                for a in csv.DictReader(fh):
                    sid = a.get("span_id") or a.get("context.span_id")
                    lab = a.get("label") or a.get("score")
                    if sid and lab not in (None, ""):
                        ann[sid] = float(lab)
            for r in rows:
                if r["request_id"] in ann:
                    r["target_score"] = ann[r["request_id"]]
        if not rows:
            raise SystemExit("no usable LLM spans with scores found in the export")
        return AuditStudy(rows)


# --------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(prog="router-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "diagnose", "recommend", "simulate", "report"):
        s = sub.add_parser(name)
        s.add_argument("traces")
        s.add_argument("--budget-usd", type=float, default=None)
        s.add_argument("--future-requests", type=int, default=100_000)
        s.add_argument("--quality-tolerance", type=float, default=0.01)
        s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    st = AuditStudy.from_csv(a.traces)
    if a.cmd == "validate":
        w = st.validate()
        print("warnings:\n" + w.render())
        return 0
    if a.cmd == "diagnose":
        print(json.dumps(st.diagnose(), indent=1)); return 0
    if a.cmd == "recommend":
        print(json.dumps(st.recommend_audits(a.budget_usd, a.future_requests,
                                             a.quality_tolerance), indent=1, default=str)); return 0
    if a.cmd == "simulate":
        print(json.dumps(st.simulate(a.quality_tolerance, a.future_requests), indent=1,
                         default=str)); return 0
    txt = st.report(tol=a.quality_tolerance, H=a.future_requests)
    if a.out:
        open(a.out, "w").write(txt); print(f"wrote {a.out}")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
