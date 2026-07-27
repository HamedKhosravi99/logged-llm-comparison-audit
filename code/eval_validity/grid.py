"""Frozen per-partition base objects for the selection-vs-measurement validity study.

One call to `build_base(seed)` draws ONE prompt-cohort partition of the LLMRouterBench
performance-cost track and freezes everything that must not depend on the evaluator condition or
on the assignment regime:

  * five prompt-disjoint cohorts   router-train 20% / log 50% / acquisition 12% /
                                   verification 8% / scoring 10% (stratified by benchmark slice);
  * J=8 contexts                   KMeans on the router-train prompt embeddings, frozen, nearest
                                   centre for every other prompt;
  * the incumbent router           a FrugalGPT-style cost cascade fit on router-train only;
  * the evaluator-distortion       model capability ranks z_k, context difficulty signs s_j and the
    design constants               bias magnitude lambda -- all computed on router-train only.

Nothing here reads the scoring cohort's outcomes, so no design constant can depend on the final
target used for scoring.
"""
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "llmrouterbench")
GRID_PKL = os.path.join(DATA, "flagship_grid.pkl")
EMB_NPY = os.path.join(DATA, "emb_minilm.npy")

COHORTS = ["train", "log", "acq", "ver", "score"]
FRACS = (0.20, 0.50, 0.12, 0.08, 0.10)
SLICES = ["simpleqa", "mmlupro", "hle", "livecodebench", "arenahard"]

# second corpus: the released RouterBench 0-shot matrix, used to replicate the whole design on a
# different benchmark suite with a different model roster (11 models rather than 5)
RB_PKL = os.path.join(HERE, "data", "routerbench_0shot.pkl")
RB_EMB = os.path.join(HERE, "data", "routerbench_tfidf_svd.npy")
RB_MODELS = ["gpt-4-1106-preview", "claude-v2", "claude-v1", "claude-instant-v1",
             "gpt-3.5-turbo-1106", "mistralai/mixtral-8x7b-chat", "meta/llama-2-70b-chat",
             "zero-one-ai/Yi-34B-Chat", "mistralai/mistral-7b-chat",
             "meta/code-llama-instruct-34b-chat", "WizardLM/WizardLM-13B-V1.2"]
RB_SLICES = ["hellaswag", "gsm8k", "mmlu", "arc-winogrande", "other"]

_RAW = {}


def _rb_slice(name):
    n = str(name).lower()
    if n.startswith("mmlu"):
        return "mmlu"
    if n == "hellaswag":
        return "hellaswag"
    if n in ("grade-school-math", "gsm8k"):
        return "gsm8k"
    if n in ("arc-challenge", "winogrande"):
        return "arc-winogrande"
    return "other"


def _rb_embeddings(prompts):
    """TF-IDF + SVD prompt embeddings, cached. RouterBench ships no released embedding, and the
    context definition must not depend on outcomes, so a text-only representation is the right one."""
    if os.path.exists(RB_EMB):
        return np.load(RB_EMB)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    X = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2),
                        sublinear_tf=True).fit_transform([(p or "")[:2000] for p in prompts])
    E = normalize(TruncatedSVD(128, random_state=0).fit_transform(X)).astype(np.float32)
    np.save(RB_EMB, E)
    return E


def load_raw(corpus="llmrouterbench"):
    """Prompt-level outcome matrix + costs + slice labels + prompt embeddings, for either corpus."""
    if corpus in _RAW:
        return _RAW[corpus]
    if corpus == "llmrouterbench":
        if not os.path.exists(GRID_PKL):
            raise FileNotFoundError(
                f"{GRID_PKL} missing -- run `python load_llmrouterbench.py` to build it from the "
                "released bench-release tree (see README for the download instructions).")
        g = pickle.load(open(GRID_PKL, "rb"))
        g["ds"] = np.asarray(g["ds"], object)
        g["E"] = np.load(EMB_NPY)
    elif corpus == "routerbench":
        import pandas as pd
        df = pd.read_pickle(RB_PKL)
        Q = df[RB_MODELS].to_numpy(float)
        C = df[[m + "|total_cost" for m in RB_MODELS]].to_numpy(float)
        ds = np.asarray([_rb_slice(v) for v in df["eval_name"].to_numpy()], object)
        prompts = [str(x) for x in df["prompt"].to_numpy()]
        g = dict(Q=Q, C=C, ds=ds, queries=prompts, models=list(RB_MODELS),
                 E=_rb_embeddings(prompts))
    else:
        raise ValueError(f"unknown corpus {corpus!r}")
    _RAW[corpus] = g
    return g


def _cohorts(ds, rng):
    """Prompt-disjoint cohort labels, stratified within each benchmark slice."""
    coh = np.empty(len(ds), object)
    for d in np.unique(ds):
        idx = np.where(ds == d)[0]
        rng.shuffle(idx)
        cut = (np.cumsum(FRACS) * len(idx)).astype(int)[:-1]
        for nm, part in zip(COHORTS, np.split(idx, cut)):
            coh[part] = nm
    return coh


def build_base(seed, n_clusters=8, casc_thresh=0.5, corpus="llmrouterbench"):
    """One frozen prompt-cohort partition. Returns a dict of arrays; see module docstring."""
    g = load_raw(corpus)
    Q, C, ds, E = g["Q"], g["C"], g["ds"], g["E"]
    N, K = Q.shape
    J = n_clusters
    rng = np.random.default_rng(seed)

    coh = _cohorts(ds, rng)
    tr = np.where(coh == "train")[0]

    # contexts: frozen on router-train, nearest centre elsewhere
    km = KMeans(J, random_state=seed, n_init=10).fit(E[tr])
    ctx = km.predict(E)

    # incumbent: FrugalGPT-style cost cascade over a kNN score predictor fit on router-train only.
    # It selects on a per-prompt difficulty signal finer than the J contexts, so it escalates hard
    # prompts to stronger models and induces genuine WITHIN-context selection.
    cnorm = C.mean(0) / C.mean()
    nn = NearestNeighbors(n_neighbors=min(50, len(tr))).fit(E[tr])
    _, ind = nn.kneighbors(E)
    scorehat = Q[tr][ind].mean(1)                       # [N, K] predicted reference score
    order = np.argsort(cnorm)                           # cheapest model first
    inc_prompt = np.full(N, -1)
    clears = scorehat >= casc_thresh
    for k in order:
        hit = clears[:, k] & (inc_prompt == -1)
        inc_prompt[hit] = int(k)
    inc_prompt[inc_prompt == -1] = int(order[-1])       # escalate the rest to the priciest model
    inc = np.array([int(np.bincount(inc_prompt[ctx == j], minlength=K).argmax()) for j in range(J)])

    # ---- evaluator-design constants, router-train cohort only -------------------------------
    mu_tr = np.zeros((K, J))
    for j in range(J):
        m = (coh == "train") & (ctx == j)
        mu_tr[:, j] = Q[m].mean(0) if m.sum() else Q[coh == "train"].mean(0)

    rank = np.argsort(np.argsort(Q[tr].mean(0))).astype(float)   # 0 = weakest model on train
    z = (rank - rank.mean()) / rank.std()                        # standardized capability rank

    d_tr = 1.0 - mu_tr.mean(0)                                   # context difficulty on train
    s = np.sign(d_tr - np.median(d_tr))
    s[s == 0] = 1.0                                              # ties -> "hard" side

    srt = np.sort(mu_tr, axis=0)[::-1]
    delta_tr = srt[0] - srt[1]                                   # per-context top-two target gap
    lam = 0.5 * float(np.median(delta_tr))                       # primary bias magnitude (fixed)

    return dict(Q=Q, C=C, ds=ds, ctx=ctx, coh=coh, K=int(K), J=int(J),
                models=g["models"], inc_prompt=inc_prompt, inc=inc, scorehat=scorehat,
                z=z, s=s, lam=lam, delta_tr=delta_tr, mu_tr=mu_tr, seed=int(seed),
                corpus=corpus)
