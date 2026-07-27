# Benchmark matrices (not redistributed)

Both corpora are released by their authors and are not copied into this repository. Place them here
with the layout below, then everything under `code/` runs unchanged.

```
code/data/
  llmrouterbench/
    bench-release/          # the released per-request JSON tree
    flagship_grid.pkl       # built by `python code/load_llmrouterbench.py`
    emb_minilm.npy          # cached MiniLM request embeddings (see below)
  routerbench_0shot.pkl     # the released RouterBench 0-shot dataframe
  routerbench_tfidf_svd.npy # written automatically on first use, no action needed
```

**LLMRouterBench** (primary corpus, 11,289 requests x 5 models). Obtain the released
performance--cost track, unpack it as `code/data/llmrouterbench/bench-release/`, then run
`python code/load_llmrouterbench.py`, which aligns requests across models and writes
`flagship_grid.pkl`. `emb_minilm.npy` holds one 128-dimensional embedding per request in the same
row order as that pickle; any sentence embedding works, and
`sentence-transformers/all-MiniLM-L6-v2` with L2 normalisation is what the reported runs used. The
context clustering is fit on the router-train cohort only, so the embedding never sees an outcome.

**RouterBench** (replication corpus, 36,497 requests x 11 models). Obtain the released 0-shot
dataframe and save it as `code/data/routerbench_0shot.pkl`. It must contain the eleven model score
columns, their `|total_cost` counterparts, `eval_name` and `prompt`. No embedding file is needed,
because `code/eval_validity/grid.py` derives TF-IDF plus SVD request embeddings on first use and
caches them next to the pickle.

Nothing in this repository performs new model inference. Both corpora are replayed offline from
outputs their authors already released.
