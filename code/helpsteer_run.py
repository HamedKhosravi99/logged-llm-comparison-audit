"""Work Package C. Correctness versus polish, on HelpSteer2.

C1  run a real judge over responses that already carry five human dimensions
C2  regress the judge score on those dimensions, asking whether correctness gets the largest weight
C3  the decision-focused outcome: how often the judge picks a different response than correctness does

The point is the paper's measurement-validity claim with a measured judge rather than a synthetic
distortion: if the judge rewards verbosity or complexity after conditioning on correctness, then an
evaluation built on it measures something other than the outcome a deployment decision is about.

    python helpsteer_run.py --dry-run --limit 200
    python helpsteer_run.py --limit 400
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60)}
PRICE_DATE = "2026-07-29"

SYSTEM = ("Rate the assistant response to the user prompt on a 0-4 integer scale. "
          "Reply with JSON only: "
          '{"overall_score":0-4,"correctness_score":0-4,"helpfulness_score":0-4,'
          '"confidence":0.0-1.0}')


def _parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: float(d[k]) for k in ("overall_score", "correctness_score", "helpfulness_score")
                if k in d} or None
    except Exception:
        return None


def run(limit=400, model="gpt-4o-mini", dry=False, seed=0, workers=16, retries=2):
    df = pd.read_parquet(os.path.join(DATA, "helpsteer2.parquet"))
    # keep prompts with more than one response, which is what C3 needs
    multi = df.groupby("prompt").filter(lambda g: len(g) > 1)
    rng = np.random.default_rng(seed)
    prompts = multi["prompt"].drop_duplicates().values
    take = prompts[rng.choice(len(prompts), size=min(limit // 2, len(prompts)), replace=False)]
    sub = multi[multi["prompt"].isin(set(take))].copy()
    pin, pout = PRICES[model]
    client = None
    if not dry:
        from openai import OpenAI
        client = OpenAI()

    def work(ix):
        i, r = ix
        rec = {d: float(r[d]) for d in DIMS}
        rec.update(idx=int(i), prompt=r["prompt"][:200],
                   resp_words=len(str(r["response"]).split()))
        p = f"### Prompt\n{r['prompt'][:2500]}\n\n### Response\n{str(r['response'])[:3500]}"
        if dry:
            rec["tok_in"] = (len(SYSTEM) + len(p)) / 4.0
            rec["tok_out"] = 30
            return rec
        for a in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": p}])
                got = _parse(resp.choices[0].message.content or "")
                u = resp.usage
                rec["tok_in"] = int(u.prompt_tokens)
                rec["tok_out"] = int(u.completion_tokens)
                if got:
                    rec.update({f"judge_{k}": v for k, v in got.items()})
                    return rec
            except Exception as e:
                rec.setdefault("errors", []).append(type(e).__name__)
                time.sleep(1.5 * (a + 1))
        rec["parse_failed"] = True
        return rec

    t0 = time.time()
    rows = list(sub.iterrows())
    if dry or workers <= 1:
        out = [work(x) for x in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(work, rows))
    tin = sum(r.get("tok_in", 0) for r in out)
    tout = sum(r.get("tok_out", 0) for r in out)
    cost = (tin * pin + tout * pout) / 1e6
    meta = dict(n=len(out), judge=model, dry_run=dry, judge_cost_usd=cost,
                price_date=PRICE_DATE, prompt_tokens=int(tin), completion_tokens=int(tout),
                parse_failures=sum(1 for r in out if r.get("parse_failed")),
                seconds=round(time.time() - t0, 1))
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "helpsteer_judge%s.json" % ("_dryrun" if dry else ""))
    json.dump(dict(meta=meta, rows=out), open(path, "w"), indent=1, default=float)
    print(json.dumps(meta, indent=1))
    if dry:
        print(f"\n  extrapolated for all {len(multi):,} multi-response rows: "
              f"${cost/len(out)*len(multi):,.2f}")
    else:
        analyse(out)
    print(f"  wrote {path}")


def analyse(rows):
    d = pd.DataFrame([r for r in rows if "judge_overall_score" in r])
    if len(d) < 30:
        print("  too few judged rows to analyse")
        return
    print(f"\nC2. Judge score regressed on the five human dimensions, n={len(d):,}")
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in DIMS])
    y = d["judge_overall_score"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = ["intercept"] + DIMS
    for n, b in zip(names, beta):
        print(f"    {n:14} {b:+.4f}")
    order = sorted(zip(DIMS, beta[1:]), key=lambda t: -abs(t[1]))
    print(f"  largest weight: {order[0][0]} ({order[0][1]:+.4f})")
    print(f"  correctness is {'' if order[0][0]=='correctness' else 'NOT '}the largest weight")
    # partial effect of verbosity and complexity given correctness
    for extra in ("verbosity", "complexity"):
        cols = ["correctness", extra]
        Xp = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        print(f"    given correctness alone, {extra} still moves the judge by {bp[2]:+.4f}")

    print("\nC3. Decision-focused: which response would each criterion pick?")
    dis, n = 0, 0
    for _, g in d.groupby("prompt"):
        if len(g) < 2:
            continue
        n += 1
        if g["judge_overall_score"].idxmax() != g["correctness"].idxmax():
            dis += 1
    if n:
        print(f"    the judge picks a different response than human correctness in "
              f"{dis}/{n} = {dis/n:.1%} of prompts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run(a.limit, a.model, a.dry_run, workers=a.workers)
