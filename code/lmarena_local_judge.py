"""Work Package B, second-evaluator arm. A local judge, run at zero API cost.

Why this exists. Every measurement claim in App. H rests on a single automatic judge
(`gpt-4o-mini`). That leaves the paper unable to say whether the position flip and the length
sensitivity are properties of *that* judge or of automatic judges in general. This runs a second,
architecturally unrelated judge over a stratified subsample of the SAME battles, so the two verdicts
are directly comparable item by item.

What is held fixed. The system prompt, the answer-order protocol, the truncation length and the
verdict parser are imported from `lmarena_judge` rather than restated, so the only difference between
the two arms is the model. Order is randomised per battle and reversed verdicts are mapped back, as
in the API arm.

This adds an arm. It does not substitute for one: no cell measured by an OpenAI model is refilled
with a local model, because that would break the roster comparison the work package rests on.
"""
import os, re, json, time, random, argparse
import numpy as np
import pandas as pd

from lmarena_judge import SYSTEM, _text, _prompt, _parse, _unreverse, MAX_CHARS


def _repair(txt):
    """One tolerated formatting repair, applied to this arm only and counted in the output.

    The 7B model sometimes copies the separator out of the system prompt's schema spec
    (`"winner":"A"|"B"|...`) and emits `{"winner":"A"|"confidence":0.8}`, which is not valid JSON
    even though the verdict itself is unambiguous. Rewriting `"|"` to `","` recovers the record
    without touching the verdict. The API arm needed no such repair, so `parse_repairs` in the
    output records how often this fired, and it must be reported alongside any comparison.
    """
    return re.sub(r'"\s*\|\s*"', '","', txt)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def out_path(model_id):
    """One file per judge. Sharing one path would make a second judge resume from the first's rows."""
    slug = model_id.split("/")[-1].replace(".", "-")
    return os.path.join(RES, f"lmarena_local_judge_{slug}.json")


def _api_verdicts():
    """battle_id -> the gpt-4o-mini record, across all three splits."""
    seen = {}
    for split in ("calibration", "validation", "test"):
        p = os.path.join(RES, f"lmarena_judge_{split}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["rows"]:
            seen[r["battle_id"]] = r
    return seen


def _stratified(df, api, n, seed):
    """Sample n battles the API judge already scored, stratified by split and code/non-code."""
    df = df[df["id"].astype(str).isin(api)].copy()
    df["is_code"] = [bool(api[str(i)].get("is_code", False)) for i in df["id"]]
    rng = random.Random(seed)
    picked, groups = [], list(df.groupby(["split", "is_code"]))
    per = max(1, n // max(1, len(groups)))
    for _, g in groups:
        rows = g.to_dict("records")
        rng.shuffle(rows)
        picked += rows[:per]
    rng.shuffle(picked)
    return picked[:n]


def run(n=600, model_id=DEFAULT_MODEL, seed=0, max_tokens=200, save_every=25, limit_hours=None):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    df = pd.read_parquet(os.path.join(DATA, "lmarena_sample.parquet"))
    api = _api_verdicts()
    if not api:
        raise SystemExit("no gpt-4o-mini result files found; run lmarena_judge.py first")
    rows = _stratified(df, api, n, seed)
    print(f"  {len(api):,} battles carry an API verdict; sampling {len(rows)}")

    OUT = out_path(model_id)
    done = {}
    if os.path.exists(OUT):                      # resumable
        for r in json.load(open(OUT))["rows"]:
            done[r["battle_id"]] = r
        print(f"  resuming, {len(done)} already judged")

    model, tok = load(model_id)
    sampler = make_sampler(temp=0.0)             # greedy, matching temperature=0 in the API arm
    out, fails, t0 = list(done.values()), 0, time.time()

    repairs = [0]

    def ask(row, reverse):
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": _prompt(row, reverse)}],
            add_generation_prompt=True, tokenize=False)
        t = time.time()
        txt = generate(model, tok, prompt=text, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        v = _parse(txt)
        if v is None:
            v = _parse(_repair(txt))
            if v is not None:
                repairs[0] += 1
        return v, time.time() - t, len(tok.encode(text))

    for i, r in enumerate(rows):
        bid = str(r["id"])
        if bid in done:
            continue
        if limit_hours and (time.time() - t0) / 3600 > limit_hours:
            print(f"  time limit reached at {len(out)} judged")
            break
        rev_first = random.Random(seed + i).random() < 0.5     # randomise presentation order
        a = api[bid]
        rec = dict(battle_id=bid, evaluation_session_id=str(r["evaluation_session_id"]),
                   split=str(r["split"]), judge_id=model_id, judge_prompt_version="v1",
                   human_winner=a.get("human_winner"), model_a=a.get("model_a"),
                   model_b=a.get("model_b"), is_code=a.get("is_code"),
                   tokens_a=a.get("tokens_a"), tokens_b=a.get("tokens_b"),
                   length_difference=a.get("length_difference"),
                   api_winner_original=a.get("judge_winner_original"),
                   api_winner_reversed=a.get("judge_winner_reversed"),
                   reversed_first=rev_first, judge_cost_usd=0.0)
        ok = True
        for tag, reverse in (("original", False), ("reversed", True)):
            v, dt, ntok = ask(r, reverse)
            if v is None:
                fails += 1
                ok = False
                rec[f"judge_winner_{tag}"] = None
                rec[f"judge_latency_{tag}"] = dt
                rec[f"tok_in_{tag}"] = ntok
                continue
            w = v["winner"] if tag == "original" else _unreverse(v["winner"])
            rec[f"judge_winner_{tag}"] = w
            rec[f"judge_confidence_{tag}"] = v["confidence"]
            rec[f"judge_latency_{tag}"] = dt
            rec[f"tok_in_{tag}"] = ntok
        rec["parse_ok"] = ok
        out.append(rec)

        if len(out) % save_every == 0:
            _save(out, model_id, fails, t0, repairs[0])
            el = time.time() - t0
            rate = el / max(1, len(out) - len(done))
            print(f"  {len(out)}/{len(rows)} judged, {rate:.1f}s/battle, "
                  f"{fails} parse failures, {el/3600:.2f}h elapsed", flush=True)

    _save(out, model_id, fails, t0, repairs[0])
    _gate(out)
    print(f"\n  wrote {OUT}: {len(out)} battles, {fails} parse failures, "
          f"{repairs[0]} formatting repairs, "
          f"{(time.time()-t0)/3600:.2f}h, $0.00")
    return out


def _gate(rows):
    """Is this judge better than a degenerate constant rule? The Qwen2.5 arm was not."""
    h = {"model_a": "A", "model_b": "B"}
    dec = [r for r in rows if r.get("human_winner") in h and r.get("judge_winner_original")]
    if not dec:
        print("  GATE: no decisive battles judged")
        return
    agree = sum(r["judge_winner_original"] == h[r["human_winner"]] for r in dec) / len(dec)
    constA = sum(h[r["human_winner"]] == "A" for r in dec) / len(dec)
    base = max(constA, 1 - constA)
    verdict = "PASS" if agree > base else "FAIL, no better than a constant rule"
    print(f"  GATE on {len(dec)} decisive battles: agreement {agree:.3f} "
          f"vs constant-rule baseline {base:.3f} -> {verdict}")


def _save(out, model_id, fails, t0, repairs=0):
    OUT = out_path(model_id)
    json.dump(dict(meta=dict(judge=model_id, backend="mlx", n=len(out),
                             parse_failures=fails, parse_repairs=repairs, wall_hours=(time.time() - t0) / 3600,
                             api_cost_usd=0.0, purpose="second-evaluator robustness arm"),
                   rows=out), open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit-hours", type=float, default=None)
    a = ap.parse_args()
    run(n=a.n, model_id=a.model, max_tokens=a.max_tokens, limit_hours=a.limit_hours)
