"""Work Package B, steps B3 to B5. Run a real automatic judge against real human preferences.

Protocol, exactly as the plan specifies.
  * model names are stripped before judging, so the judge cannot recognise a brand;
  * every battle is judged in the original AND reversed answer order, and reversed verdicts are
    mapped back to original answer identities, which is what makes position bias measurable;
  * output is structured and parse failures are logged, never silently dropped;
  * battle order is randomised;
  * judge cost and latency are recorded per call;
  * the frozen test ids are never used for any tuning.

    python lmarena_judge.py --dry-run --limit 200      # no API calls, prices the real run
    python lmarena_judge.py --limit 50                 # small real pilot
    python lmarena_judge.py --split test               # the full arm
"""
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))

SYSTEM = ("You are comparing two assistant responses to the same user prompt. "
          "Judge only which response better serves the user. "
          "Reply with JSON only: "
          '{"winner":"A"|"B"|"tie"|"both_bad","confidence":0.0-1.0,"reason":"one short sentence"}')

# per 1M tokens, in / out. Frozen here with a date so the cost line is reproducible.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60),
          "gpt-4.1-nano": (0.10, 0.40), "gpt-4o": (2.50, 10.00)}
PRICE_DATE = "2026-07-29"
MAX_CHARS = 4000          # truncate very long conversations symmetrically


def _text(conv):
    try:
        turns = conv if isinstance(conv, (list, np.ndarray)) else json.loads(conv)
        return "\n".join(f"{t.get('role','')}: {t.get('content','')}" for t in turns)
    except Exception:
        return str(conv)


def _prompt(row, reverse):
    a, b = _text(row["conversation_a"]), _text(row["conversation_b"])
    if reverse:
        a, b = b, a
    return f"### Response A\n{a[:MAX_CHARS]}\n\n### Response B\n{b[:MAX_CHARS]}"


def _parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    w = str(d.get("winner", "")).strip().lower()
    if w in ("a", "model_a"):
        w = "A"
    elif w in ("b", "model_b"):
        w = "B"
    elif w not in ("tie", "both_bad"):
        return None
    try:
        c = float(d.get("confidence", 0.5))
    except Exception:
        c = 0.5
    return dict(winner=w, confidence=max(0.0, min(1.0, c)), reason=str(d.get("reason", ""))[:200])


def _unreverse(w):
    """Map a verdict given on the reversed presentation back to original answer identities."""
    return {"A": "B", "B": "A"}.get(w, w)


def run(split="calibration", limit=None, model="gpt-4o-mini", dry=False, seed=0, retries=2,
        workers=12):
    df = pd.read_parquet(os.path.join(DATA, "lmarena_sample.parquet"))
    df = df[df["split"] == split]
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]

    client = None
    if not dry:
        from openai import OpenAI
        client = OpenAI()

    pin, pout = PRICES[model]
    out, fails, tin, tout, t0 = [], 0, 0, 0, time.time()

    def work(ir):
        i, r = ir
        rec = dict(battle_id=str(r["id"]), evaluation_session_id=str(r["evaluation_session_id"]),
                   split=split, judge_id=model, judge_prompt_version="v1",
                   human_winner=r["winner"], model_a=r["model_a"], model_b=r["model_b"],
                   is_code=bool(r["is_code"]), tokens_a=int(r["tok_a"]), tokens_b=int(r["tok_b"]),
                   length_difference=int(r["len_diff"]))
        for tag, rev in (("original", False), ("reversed", True)):
            p = _prompt(r, rev)
            est_in = (len(SYSTEM) + len(p)) / 4.0
            if dry:
                rec[f"tok_in_{tag}"] = est_in
                rec[f"tok_out_{tag}"] = 40
                rec[f"judge_winner_{tag}"] = None
                rec[f"judge_cost_{tag}"] = (est_in * pin + 40 * pout) / 1e6
                continue
            got, lat = None, None
            for attempt in range(retries + 1):
                try:
                    s = time.time()
                    resp = client.chat.completions.create(
                        model=model, temperature=0,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": p}])
                    lat = time.time() - s
                    got = _parse(resp.choices[0].message.content or "")
                    u = resp.usage
                    rec[f"tok_in_{tag}"] = int(u.prompt_tokens)
                    rec[f"tok_out_{tag}"] = int(u.completion_tokens)
                    rec[f"judge_cost_{tag}"] = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
                    if got:
                        break
                except Exception as e:
                    rec.setdefault("errors", []).append(f"{tag}:{type(e).__name__}")
                    time.sleep(1.5 * (attempt + 1))
            if got is None:
                rec[f"parse_failed_{tag}"] = True
                rec[f"judge_winner_{tag}"] = None
            else:
                w = got["winner"]
                rec[f"judge_winner_{tag}"] = _unreverse(w) if rev else w
                rec[f"judge_confidence_{tag}"] = got["confidence"]
                rec[f"judge_latency_{tag}"] = lat
        return rec

    if dry or workers <= 1:
        for ir in enumerate(rows):
            out.append(work(ir))
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(work, list(enumerate(rows))):
                out.append(rec)
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(rows)}  [{time.time()-t0:.0f}s]  "
                          f"${(tin*pin+tout*pout)/1e6:.3f}", flush=True)

    tin = sum(r.get(f"tok_in_{t}", 0) for r in out for t in ("original", "reversed"))
    tout = sum(r.get(f"tok_out_{t}", 0) for r in out for t in ("original", "reversed"))
    fails = sum(1 for r in out for t in ("original", "reversed") if r.get(f"parse_failed_{t}"))
    api_errors = sum(len(r.get("errors", [])) for r in out)
    cost = (tin * pin + tout * pout) / 1e6
    meta = dict(split=split, n_battles=len(rows), judge=model, dry_run=dry,
                price_per_1m_in=pin, price_per_1m_out=pout, price_date=PRICE_DATE,
                prompt_tokens=int(tin), completion_tokens=int(tout),
                judge_cost_usd=cost, parse_failures=fails, api_errors=api_errors,
                seconds=round(time.time() - t0, 1), workers=workers)
    os.makedirs(RES, exist_ok=True)
    tag = "dryrun" if dry else split
    path = os.path.join(RES, f"lmarena_judge_{tag}.json")
    json.dump(dict(meta=meta, rows=out), open(path, "w"), indent=1)
    print(json.dumps(meta, indent=1))
    if dry:
        full = len(pd.read_parquet(os.path.join(DATA, "lmarena_sample.parquet")))
        print(f"\n  extrapolated cost for all {full:,} battles, both orders, one judge: "
              f"${cost / len(rows) * full:,.2f}")
    print(f"  wrote {path}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run(a.split, a.limit, a.model, a.dry_run, workers=a.workers)
