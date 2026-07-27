"""Load the LLMRouterBench performance-cost track into an aligned (prompt x model) grid.

For each of the 5 flagship datasets and 5 flagship models we read the released per-prompt JSON
(fields: index, score, cost, prompt_tokens, completion_tokens, origin_query) and align prompts
across models by `index`, keeping only prompts scored by all 5 models. Output (cached to npz):
  Q  [N,K] float  per-prompt score in [0,1]
  C  [N,K] float  per-prompt dollar cost
  ds [N]   object dataset label (routing sub-population marker; embeddings give the contexts)
  queries [N] list of origin_query strings (for embedding-based contexts)
Offline replay only -- these are the benchmark's released outputs, no new inference.
"""
import os, glob, json, numpy as np, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "llmrouterbench", "bench-release")
CACHE = os.path.join(HERE, "data", "llmrouterbench", "flagship_grid.pkl")

MODELS = ["gemini-2.5-flash", "deepseek-v3-0324", "qwen3-235b-a22b-thinking-2507",
          "gpt-5-chat", "claude-sonnet-4"]
# dataset -> split subdir (None = models sit directly under the dataset dir, as in arenahard)
DATASETS = {"simpleqa": "test", "mmlupro": "test_3000", "hle": "test",
            "livecodebench": "test", "arenahard": None}


def _model_json(ds, split, model):
    base = os.path.join(ROOT, ds, split, model) if split else os.path.join(ROOT, ds, model)
    fs = sorted(glob.glob(os.path.join(base, "*.json")))
    return fs[0] if fs else None


def load(datasets=DATASETS, models=MODELS, verbose=True):
    Q_rows, C_rows, ds_rows, q_rows = [], [], [], []
    for ds, split in datasets.items():
        recs = {}
        for m in models:
            f = _model_json(ds, split, m)
            if not f:
                raise FileNotFoundError(f"{ds}/{split}/{m} json missing")
            d = json.load(open(f))["records"]
            recs[m] = {r["index"]: (float(r["score"]), float(r["cost"]), r.get("origin_query", "") or r.get("prompt", ""))
                       for r in d}
        common = sorted(set.intersection(*[set(recs[m]) for m in models]))
        for idx in common:
            Q_rows.append([recs[m][idx][0] for m in models])
            C_rows.append([recs[m][idx][1] for m in models])
            q_rows.append(recs[models[0]][idx][2])
            ds_rows.append(ds)
        if verbose:
            print(f"{ds:14s} split={split!s:10s} aligned prompts: {len(common)}", flush=True)
    Q = np.asarray(Q_rows, float); C = np.asarray(C_rows, float)
    ds_arr = np.asarray(ds_rows, object); queries = q_rows
    if verbose:
        print(f"TOTAL prompts={Q.shape[0]}  models={Q.shape[1]}", flush=True)
        print("mean score per model:", dict(zip(models, np.round(Q.mean(0), 3))))
        print("mean $cost per model:", dict(zip(models, np.round(C.mean(0), 5))))
    return dict(Q=Q, C=C, ds=ds_arr, queries=queries, models=models)


if __name__ == "__main__":
    g = load()
    pickle.dump(g, open(CACHE, "wb"))
    print("cached ->", CACHE, flush=True)
