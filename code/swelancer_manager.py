"""Work Package E. Dollar-valued work, SWE-Lancer managerial tasks.

E1  the model chooses among real implementation proposals; the target is the decision the hired
    engineering manager actually made
E3  the human-fallback scenario: a wrong choice costs the task's released payout

This subset needs no container, unlike the independent-task split, because the whole task is
reading an issue and its proposals and picking one. Ground truth is `game.correct_proposal.id`,
present on all 265 managerial tasks, whose released payouts total $264,500.

    python swelancer_manager.py --dry-run --limit 20
    python swelancer_manager.py --limit 120 --models gpt-4.1-nano,gpt-4o-mini,gpt-4.1-mini,gpt-4o
"""
import argparse, ast, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
CSV = os.path.abspath(os.path.join(
    HERE, "..", "ext", "frontier-evals", "project", "swelancer", "all_swelancer_tasks.csv"))
PRICES = {"gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60),
          "gpt-4.1-mini": (0.40, 1.60), "gpt-4o": (2.50, 10.00)}
PRICE_DATE = "2026-07-29"
SYSTEM = ("You are the engineering manager for this issue. Exactly one proposal was ultimately "
          "implemented and merged. Choose it. Reply with JSON only: "
          '{"proposal_id": <int>, "confidence": 0.0-1.0}')
MAXC = 2200


def load(limit=None, seed=0):
    df = pd.read_csv(CSV)
    df = df[df["variant"] == "swe_manager"].copy()
    out = []
    for _, r in df.iterrows():
        try:
            md = ast.literal_eval(r["manager_data"])
            g = md["game"]
            props = g["all_formal_proposals"]
            cid = int(g["correct_proposal"]["id"])
            if not props or all(int(p["id"]) != cid for p in props):
                continue
            out.append(dict(question_id=r["question_id"], price=float(r["price"]),
                            title=str(md.get("title", ""))[:200],
                            issue=str(md.get("issue_repo_steps", ""))[:1500],
                            proposals=[(int(p["id"]), str(p["content"])[:MAXC]) for p in props],
                            correct_id=cid))
        except Exception:
            continue
    if limit:
        rng = np.random.default_rng(seed)
        # stratify by price band so the dollar-weighted estimate is not dominated by one band
        df2 = pd.DataFrame(out)
        df2["band"] = pd.cut(df2["price"], [0, 500, 1000, 2000, 1e9], labels=False)
        pick = []
        for b, g in df2.groupby("band"):
            k = max(1, int(round(limit * len(g) / len(df2))))
            pick += list(rng.choice(g.index.values, size=min(k, len(g)), replace=False))
        out = [out[i] for i in sorted(pick)]
    return out


def _prompt(t):
    body = "\n\n".join(f"### Proposal {i}\n{c}" for i, c in t["proposals"])
    return f"## Issue\n{t['title']}\n{t['issue']}\n\n{body}\n\nWhich proposal id was implemented?"


def _parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d["proposal_id"]), float(d.get("confidence", 0.5))
        except Exception:
            pass
    m = re.search(r"\b(\d+)\b", txt or "")
    return (int(m.group(1)), 0.5) if m else (None, 0.0)


def run(models, limit=120, dry=False, workers=12, retries=2, seed=0):
    tasks = load(limit, seed)
    print(f"{len(tasks)} managerial tasks, released value ${sum(t['price'] for t in tasks):,.0f}")
    client = None
    if not dry:
        from openai import OpenAI
        client = OpenAI()
    rows = []
    for model in models:
        pin, pout = PRICES[model]

        def work(t):
            rec = dict(question_id=t["question_id"], task_value_usd=t["price"],
                       model_id=model, correct_id=t["correct_id"], n_proposals=len(t["proposals"]))
            p = _prompt(t)
            if dry:
                rec.update(tok_in=(len(SYSTEM) + len(p)) / 4.0, tok_out=20,
                           selected_proposal=None, correct_selection=None)
                return rec
            for a in range(retries + 1):
                try:
                    s = time.time()
                    r = client.chat.completions.create(
                        model=model, temperature=0,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": p}])
                    sel, conf = _parse(r.choices[0].message.content or "")
                    u = r.usage
                    rec.update(tok_in=int(u.prompt_tokens), tok_out=int(u.completion_tokens),
                               model_cost_usd=(u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6,
                               selected_proposal=sel, confidence=conf,
                               correct_selection=bool(sel == t["correct_id"]),
                               latency_ms=int(1000 * (time.time() - s)))
                    if sel is not None:
                        return rec
                except Exception as e:
                    rec.setdefault("errors", []).append(type(e).__name__)
                    time.sleep(1.5 * (a + 1))
            rec.setdefault("correct_selection", False)
            rec["parse_failed"] = True
            return rec

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1 if dry else workers) as ex:
            got = list(ex.map(work, tasks))
        rows += got
        cost = sum(r.get("model_cost_usd", 0) for r in got)
        if not dry:
            acc = np.mean([r["correct_selection"] for r in got])
            dw = (sum(r["task_value_usd"] * r["correct_selection"] for r in got)
                  / sum(r["task_value_usd"] for r in got))
            print(f"  {model:14} accuracy {acc:.3f}  dollar-weighted {dw:.3f}  "
                  f"model cost ${cost:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    meta = dict(n_tasks=len(tasks), models=list(models), dry_run=dry, price_date=PRICE_DATE,
                total_task_value_usd=float(sum(t["price"] for t in tasks)),
                total_model_cost_usd=float(sum(r.get("model_cost_usd", 0) for r in rows)))
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "swelancer_manager%s.json" % ("_dryrun" if dry else ""))
    json.dump(dict(meta=meta, rows=rows), open(path, "w"), indent=1, default=float)
    if dry:
        tin = sum(r["tok_in"] for r in rows); tout = sum(r["tok_out"] for r in rows)
        est = sum((sum(r["tok_in"] for r in rows if r["model_id"] == m) * PRICES[m][0]
                   + sum(r["tok_out"] for r in rows if r["model_id"] == m) * PRICES[m][1]) / 1e6
                  for m in models)
        print(f"  dry run: {tin:,.0f} in / {tout:,.0f} out tokens, estimated ${est:.3f}")
    else:
        fallback(rows)
    print(f"  wrote {path}")


def fallback(rows):
    """E3. Human-fallback economics. A wrong choice pays the released task payout."""
    df = pd.DataFrame(rows)
    print("\nE3. Model-plus-human-fallback scenario, benchmark-implied")
    human_only = df.groupby("model_id")["task_value_usd"].sum().iloc[0]
    print(f"  human-only baseline, every task paid out      ${human_only:,.0f}")
    print(f"  {'model':14}{'success':>9}{'model cost':>12}{'fallback':>12}{'total':>12}{'implied saving':>16}")
    for m, g in df.groupby("model_id"):
        mc = g["model_cost_usd"].sum()
        fb = g.loc[~g["correct_selection"].astype(bool), "task_value_usd"].sum()
        tot = mc + fb
        print(f"  {m:14}{g['correct_selection'].mean():9.3f}{mc:12.4f}{fb:12,.0f}"
              f"{tot:12,.0f}{human_only-tot:16,.0f}")
    print("\n  Released payouts are used as a fallback-cost scenario, not as realised savings.")
    print("\n  By task-value band")
    df["band"] = pd.cut(df["task_value_usd"], [0, 500, 1000, 2000, 1e9],
                        labels=["<=500", "501-1000", "1001-2000", ">2000"])
    for m, g in df.groupby("model_id"):
        s = g.groupby("band", observed=True)["correct_selection"].mean()
        print(f"    {m:14} " + "  ".join(f"{b}:{v:.2f}" for b, v in s.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-4.1-nano,gpt-4o-mini,gpt-4.1-mini,gpt-4o")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run(a.models.split(","), a.limit, a.dry_run, a.workers)
