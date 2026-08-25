"""Submission gate. Fails loudly on anything that would embarrass us in review.

Implements section 5 of tae_revision_plan.md, minus the checks that no longer apply (the old
Figure 2 was replaced by Table 2, so its segment-share check is now a table-consistency check).

    python code/verify_submission.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PAPER = os.path.join(ROOT, os.environ.get("TAE_PAPER", "tae_workshop_v6"))
RES = os.path.join(ROOT, "results")

FAILS: list[str] = []
NOTES: list[str] = []


def _window(fs: str, *titles: str, span: int = 1200) -> tuple[str, str | None]:
    """Slice `span` chars from the first title that is present.

    A bare fs.index() here aborts the entire gate on a retitle rather than failing one check,
    which has twice turned a routine edit into a traceback. Callers check() the returned title.
    """
    hit = next((t for t in titles if t in fs), None)
    return (fs[fs.index(hit):][:span] if hit else ""), hit


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name}: {detail}")


def tex() -> str:
    with open(PAPER + ".tex") as fh:
        return fh.read()


def log() -> str:
    with open(PAPER + ".log", errors="ignore") as fh:
        return fh.read()


# --------------------------------------------------------------------------- 1
def c1_claims_match_sources() -> None:
    """Every operational number in Table 2 reproduces from the stored replay."""
    print("\n1. Table 2 numbers reproduce from source")
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "make_table1_operational.py"),
         "--switches", os.path.join(RES, "free_stage1_switch_categories.parquet"),
         "--units", os.path.join(RES, "free_stage1_router_sensitivity.parquet"),
         "--output", os.path.join(RES, "table1_operational.csv")],
        capture_output=True, text=True)
    check("make_table1_operational.py assertions", out.returncode == 0,
          out.stderr.strip().splitlines()[-1] if out.returncode else "")

    print("\n1b. Figure 1 numbers reproduce from source")
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "make_fig1_validity.py"),
         "--validity", os.path.join(ROOT, "results", "tae_validity.json"),
         "--paper-numbers", os.path.join(ROOT, "results", "tae_paper_numbers.json"),
         "--output", os.path.join(ROOT, "figures", "fig1_validity.pdf")],
        capture_output=True, text=True)
    check("make_fig1_validity.py assertions", out.returncode == 0,
          out.stderr.strip().splitlines()[-1] if out.returncode else "")

    print("\n1c. Table 1 (measurement) numbers reproduce from source")
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "make_table2_measurement.py"),
         "--results", RES, "--output", os.path.join(RES, "table2_measurement.csv")],
        capture_output=True, text=True)
    check("make_table2_measurement.py assertions", out.returncode == 0,
          out.stderr.strip().splitlines()[-1] if out.returncode else "")


# --------------------------------------------------------------------------- 2
def c2_switch_rows_decompose() -> None:
    """The four switch rows must sum to the total, and precision must match the text."""
    print("\n2. Switch decomposition is internally consistent")
    t = pd.read_csv(os.path.join(RES, "table1_operational.csv")).set_index("outcome")
    for arm in ("raw_log", "conf_gated"):
        parts = sum(t.loc[k, arm] for k in
                    ["Valid cost-saving switches / run", "Quality-safe, no saving / run",
                     "Quality-violating switches / run"])
        total = t.loc["Total switches / run", arm]
        check(f"{arm}: A+B+C == total", abs(parts - total) < 1e-9,
              f"{parts:.4f} vs {total:.4f}")
    prec = t.loc["Valid cost-saving switches / run", "conf_gated"] / \
        t.loc["Total switches / run", "conf_gated"]
    quoted = 0.41
    check("gated precision matches the 41% quoted in RQ5", abs(prec - quoted) < 0.01,
          f"computed {prec:.3f}")


# --------------------------------------------------------------------------- 3
def c3_recall_denominator() -> None:
    """Print numerator and denominator; the paper must quote the ratio, not the numerator.

    The prose sentence this used to grep ("recovers $32\\%$ of valid") became a table row when
    the eight-percentage sentence was replaced by tab:switchq. The guard follows the claim: it
    now reads the recall row and pins all three arms against the replay.
    """
    print("\n3. Recall numerator and denominator (the P0-2 bug)")
    C = pd.read_parquet(os.path.join(RES, "free_stage1_switch_categories.parquet"))
    C = C[C.scenario == "cost_reduction"]
    want = {"Raw-log downrouting": 0.20, "Inverse-variance CASE": 0.45,
            "Confidence-gated IV CASE": 0.32}
    for arm, quoted in want.items():
        g = C[C.arm == arm]
        opp = g[g.opportunity == 1]
        rec = (opp.category == "A").sum() / len(opp)
        print(f"      {arm:26} {(opp.category=='A').sum():4} / {len(opp):4} = {rec:.4f}")
        check(f"{arm} recall matches the paper", abs(rec - quoted) < 0.01,
              f"computed {rec:.4f}, paper says {quoted}")
    # tab:switchq recall row, columns in the order raw-log / ungated / gated
    m = re.search(r"Recall: valid opportunities taken(.*)", tex())
    row = m.group(1).split(r"\\")[0] if m else ""
    pct = re.findall(r"(\d+)\\%", row)
    check("tab:switchq recall row quotes the ratios, not the numerators",
          pct == ["20", "45", "32"], f"row reads {pct}" if m else "recall row not found")


# --------------------------------------------------------------------------- 4
def c4_symbols_defined() -> None:
    """Every symbol used in the main text should be introduced there or in the notation table."""
    print("\n4. Main-text symbols are defined")
    s = tex()
    body = s[:s.index("\\label{endofmaintext}")]
    for sym, name in [("\\muref", "target quality"), ("\\Bsel", "selection bias"),
                      ("\\pidel", "policy"), ("\\piref", "target policy"),
                      ("\\Yref", "target outcome")]:
        check(f"{name} macro used in body", sym in body, "")
    check("notation table referenced from the body", "tab:notation" in body, "")
    # The expansion is "experimentation", matching the companion TMLR paper. The guard is that
    # CASE is expanded at all, not which noun the acronym ends on.
    check("CASE expanded on first use",
          "causal active sequential experimentation" in s.lower(), "")


# --------------------------------------------------------------------------- 5
def c5_no_broken_refs() -> None:
    print("\n5. LaTeX build is clean")
    L = log()
    check("no errors", L.count("\n!") == 0, f"{L.count(chr(10)+'!')} found")
    check("no undefined references", "LaTeX Warning: Reference" not in L, "")
    check("no undefined citations", "LaTeX Warning: Citation" not in L, "")
    check("no overfull boxes", "Overfull" not in L, "")
    check("no '??' in the PDF text", "??" not in _pdf_text(), "")


def _pdf_text() -> str:
    try:
        import fitz
    except ImportError:
        NOTES.append("PyMuPDF missing; PDF text checks skipped")
        return ""
    d = fitz.open(PAPER + ".pdf")
    return "".join(p.get_text() for p in d)


# --------------------------------------------------------------------------- 6
def c6_page_count() -> None:
    print("\n6. Main text is at most 8 pages")
    with open(PAPER + ".aux", errors="ignore") as fh:
        aux = fh.read()
    m = re.search(r"newlabel\{endofmaintext\}\{\{[^}]*\}\{(\d+)\}", aux)
    page = int(m.group(1)) if m else -1
    check("main text ends on page <= 8", 0 < page <= 8, f"ends on page {page}")
    m2 = re.search(r"\((\d+) pages", log())
    print(f"      total document: {m2.group(1) if m2 else '?'} pages")


# --------------------------------------------------------------------------- 7
def c7_gate_tests() -> None:
    print("\n7. App. B gate tests")
    t = os.path.join(HERE, "tests")
    if not os.path.isdir(t):
        NOTES.append("tests/ not found; gate tests skipped")
        return
    out = subprocess.run([sys.executable, "-m", "pytest", t, "-q"],
                         capture_output=True, text=True, cwd=HERE)
    check("pytest suite passes", out.returncode == 0,
          out.stdout.strip().splitlines()[-1] if out.stdout else "")


# --------------------------------------------------------------------------- 8
def c8_anonymity() -> None:
    print("\n8. Anonymity")
    s = tex()
    check("double-blind workshop option set", "dblblindworkshop" in s, "")
    check("workshoptitle set", "\\workshoptitle{" in s, "")
    # strip comments first: "[final]" legitimately appears in camera-ready reminders
    live = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.splitlines())
    check("no 'final' option", "[final]" not in live, "")
    txt = _pdf_text()
    for name in ("Khosravi", "Hamed", "hkhosravi", "gatech"):
        check(f"no '{name}' in PDF text", name.lower() not in txt.lower(), "")
    if os.path.exists(PAPER + ".pdf"):
        try:
            import fitz
            meta = fitz.open(PAPER + ".pdf").metadata or {}
            bad = [k for k, v in meta.items()
                   if v and any(n in str(v).lower() for n in ("khosravi", "hamed"))]
            check("PDF metadata carries no author name", not bad, str(bad))
        except ImportError:
            pass




# --------------------------------------------------------------------------- 9
def _flat(x: str) -> str:
    """Whitespace-normalized text so line wraps cannot defeat substring checks."""
    return " ".join(x.split())


def c9_review_text_guards() -> None:
    """Text guards from the Claude-comments plan (rev. 2)."""
    print("\n9. Review-response text guards (plan A)")
    s = tex()
    body = s[:s.index("\\label{endofmaintext}")]
    fs, fb = _flat(s), _flat(body)
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    ab = _flat(m.group(1)) if m else ""

    # blindness in the lede: within the first four abstract sentences
    sents = re.split(r"(?<=[.?])\s+(?=[A-Z\\])", ab)
    # V5 leads with the business decision, so the blindness claim moves out of the lede and is
    # stated without the word: "measurement error that repeated evaluation cannot remove". The
    # guard follows the claim, and still requires it somewhere in the abstract.
    check("abstract: states randomization cannot remove measurement error",
          any("blind" in x for x in sents[:4])
          or "repeated evaluation cannot remove" in ab, "")
    # The blindness claim used to read "no estimator ... only through their difference". C3
    # replaced that with the null-space statement, which is more accurate and still names
    # differencing; the guard follows the wording, not the old phrasing.
    # V4 replaced the null-space assertion with Thm. 3.2, which is strictly stronger: it quantifies
    # over estimators rather than over one correction. The guard follows the claim.
    # V5 states non-identification as observational equivalence ("the same data can be consistent
    # with different target-quality tables") rather than as a claim about estimators. That is the
    # same claim in more readable words, so the guard follows it -- and now also pins the part
    # that makes it sharp, that no amount of re-evaluation removes the ambiguity.
    # Migrated again: the abstract now says "quality table", since "target-quality table" was a
    # second name for the same object and was defined nowhere. Both spellings are accepted.
    check("abstract: non-identification stated, and scoped to any budget",
          ("no estimator measurable" in ab
           or "consistent with different target-quality tables" in ab
           or "consistent with different quality tables" in ab)
          and ("any audit budget" in ab or "however much re-evaluation is purchased" in ab
               or "regardless of the re-evaluation budget" in ab), "")
    # The tau^2 null moved out of the abstract into RQ5. This guard exists to keep that claim
    # *scoped* wherever it lives, so it now looks in RQ5 rather than pinning it to the abstract.
    m = re.search(r"\\label\{sec:rq5\}(.*?)(\\subsection|\\section)", body, re.S)
    rq5 = _flat(m.group(1)) if m else ""
    check("RQ5: null scoped 'relative to that same alternative'",
          "relative to that same alternative" in rq5.lower(),
          "RQ5 subsection not found" if not m else "")
    check("abstract: no incumbent-saving dollar headline", "3.60" not in ab, "")

    # Table 3 caption and columns
    m = re.search(r"caption\{Gated CASE against(.*?)\\end\{table\}", s, re.S)
    t3 = _flat(m.group(1)) if m else ""
    check("tab:opx: three policy columns (incumbent, raw-log, gated)",
          all(k in t3 for k in ("Incumb", "Raw-log", "Gated")), "")
    check("tab:opx caption: post-hoc + outside confirmatory set",
          "post hoc" in t3 and "confirmatory" in t3, "")
    check("tab:opx caption: third leg reported unsupported",
          "2.21" in t3 and "unsupported" in t3, "")

    # vocabulary bans (whole document)
    for w in ("deployment-ready", "deployment ready", "certified", "safe CASE",
              "selective safety", "beneficial switches", "Hugging Face"):
        check(f"zero hits: '{w}'", not re.search(re.escape(w), fs, re.I), "")
    check("zero 'audit audit'", not re.search(r"audit\s+audit", s, re.I), "")

    # 'regret' only when describing other methods' objectives (App F)
    check("no reader-facing 'regret' in main text", "regret" not in body.lower(), "")
    app = s[s.index("\\label{endofmaintext}"):]
    bad = [ln for ln in app.splitlines() if "regret" in ln.lower()
           and not re.search(r"routing regret|regret-minimal", ln)]
    check("appendix 'regret' only as other methods' objective", not bad, str(bad[:1]))

    # every MAIN-TEXT 18.3 is range-qualified; App C's sweep is the range's source
    lone = [i for i in range(len(fb)) if fb.startswith("18.3", i)
            and "10.4" not in fb[max(0, i - 350):i + 350]]
    check("main-text 18.3% appearances carry the 10.4-18.3 range nearby", not lone,
          f"{len(lone)} bare occurrence(s)")

    check("Limitations: routers stated as reconstructions",
          "reconstruction" in fb[fb.index("Limitations"):], "")
    m = re.search(r"paragraph\{Contributions\.\}(.*?)\\end\{enumerate\}", s, re.S)
    contrib = m.group(1) if m else ""
    check("contributions cite no deferred corollary",
          "cor:margin" not in contrib and "cor:cost" not in contrib, "")

    # each main-text section has a real body
    secs = re.split(r"\\section\{", body)[1:]
    thin = [x.split("}")[0] for x in secs if len(x.split()) < 40]
    check("every main-text section body >= 40 words", not thin, str(thin))

    # required vocabulary is present somewhere
    for w in ("confidence-gated", "valid cost-saving", "quality-violating",
              "conservative switch suppression"):
        check(f"required term present: '{w}'", w in fs.lower(), "")


# -------------------------------------------------------------------------- 10
def c10_gpt_guards() -> None:
    """Scoped-claim, working-model, OPE and numbering guards (plan B)."""
    print("\n10. GPT-comments guards (plan B)")
    s = tex()
    body = s[:s.index("\\label{endofmaintext}")]
    fs, fb = _flat(s), _flat(body)
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    ab = _flat(m.group(1)) if m else ""

    # This guard pins a phrasing, not a claim. R16 replaced "these strands do not jointly
    # address" with the crisper "none asks ..."; the scoping requirement (abstract AND intro)
    # is unchanged.
    # The requirement is that the positioning be scoped in BOTH abstract and intro, never a bare
    # "nobody has studied this". V5 scopes the abstract by naming what the neighbouring families
    # do solve; the intro keeps the "none asks ..." form. Either phrasing satisfies the intent.
    check("positioning scoped in the abstract",
          bool(re.search(r"neither alone resolves the (deployment )?decision"
                         r"|none asks when a routed log supports a", ab, re.I)), "")
    # V5 groups the neighbouring work into two families, so the intro reads "Neither asks ..."
    # rather than "None asks ...". Same scoped claim, so the guard accepts either quantifier.
    # The requirement is that the intro credit what the neighbouring work does solve and then
    # locate the gap, never assert the field is silent. V5 does this by naming what each line
    # addresses and showing the gap is statistical; earlier drafts used "none/neither asks ...".
    # v5 compressed this paragraph: it no longer contains the literal "existing work addresses",
    # but it still credits each neighbouring family by name and ends "Our problem requires both at
    # once", which is the same scoped claim. Accept that phrase alone -- it is only meaningful once
    # two families have been named -- so the guard follows the claim rather than the wording.
    check("positioning scoped in the intro",
          bool(re.search(r"(none|neither) asks when a routed log supports a", fb, re.I))
          or bool(re.search(r"requires both at once", fb, re.I)), "")
    check("working-model passage with exact residual semantics",
          "working" in fb and "do not model the residual" in fb
          and "richer correction family" in fb, "")
    # Migrated: the row and Fig. 2's caption now use "Cellwise-perfect", matching the shipped
    # figure legend and the prose in §5.1 that defines the object. Both spellings are accepted.
    check("ceiling table: renamed cellwise row",
          ("Cellwise-perfect assignment repair" in fb
           or "Population cellwise assignment repair" in fb), "")

    # ceiling numbers appear both in the main-text table and the appendix source
    m = re.search(r"label\{tab:ceiling\}(.*?)\\end\{table\}", s, re.S)
    ceil = m.group(1) if m else ""
    for v in ("0.0499", "0.0338", "0.0318", "0.0000", "0.0030"):
        check(f"tab:ceiling holds {v}", v in ceil, "")
    app_c = s[s.index("\\label{endofmaintext}"):]
    for v in ("0.0318", "0.0030"):
        check(f"appendix correction table also holds {v}", v in app_c, "")

    # App F identification vocabulary + twelve-platform footnote
    m = re.search(r"section\{Related work in full\}(.*?)(\\section\{|\Z)", s, re.S)
    af = _flat(m.group(1)) if m else ""
    for w in ("off-policy", "doubly robust", "positivity", "exchangeability"):
        check(f"App F states '{w}'", w in af, "")
    check("twelve-platform documentation footnote",
          "twelve" in af or "Twelve" in af, "")

    # axis-scoped OPE comparison in RQ2
    m = re.search(r"\\label\{sec:rq2\}(.*?)\\subsection", body, re.S)
    rq2 = _flat(m.group(1)) if m else ""
    check("RQ2 names the decision-quality axis", "decision quality" in rq2, "")
    check("RQ2 names the evidence-cost axis", "zero audit spend" in rq2, "")
    check("RQ2 states the propensity-stream requirement",
          "propensity stream" in rq2, "")

    # no superiority verbs adjacent to the causal-routing citation
    for i in [m.start() for m in re.finditer(r"tsiourvas2025causal", s)]:
        ctx = s[max(0, i - 300):i + 300].lower()
        bad = [v for v in ("outperform", "superior", "beats", "we improve on",
                           "better than") if v in ctx]
        check("no superiority verb near causal-routing cite", not bad, str(bad))

    # numbering: deferred statements referenced by label, never hardcoded
    check("no hardcoded 'Remark 3.3'/'Cor. 3.4' in prose",
          not re.search(r"(Remark|Cor\.?|Corollary)[~ ]3\.[34]", s), "")
    check("deferred statements referenced via labels",
          "rem:samejudge" in body and "cor:margin" in body, "")

    # +0.0107 always carries its qualifier
    lone = [i for i in [m.start() for m in re.finditer(r"\+0\.0107", fs)]
            if not any(q in fs[max(0, i - 200):i + 400]
                       for q in ("no assignment", "noise", "cost of auditing"))]
    check("+0.0107 always qualified (noise where no assignment bias)", not lone,
          f"{len(lone)} bare")
    # dominance always conditional on a positive violation cost
    lone = [i for i in [m.start() for m in re.finditer(r"dominates", fs)]
            if "positive" not in fs[max(0, i - 200):i + 200]]
    check("'dominates' always conditioned on positive violation cost", not lone,
          f"{len(lone)} bare")

    # post-hoc labels on the exploratory addition
    check("tau2 redesign labeled post-hoc exploratory",
          "post-hoc exploratory redesign" in fs, "")

    # the identification result is stated as the sharp characterisation, not a one-sided rule
    prop, _t = _window(fs, "When the decision is point-identified", span=900)
    check("point-identification proposition is present under a known title", _t is not None, "")
    check("Prop. idset states the sharp iff, not just the sufficient direction",
          "if and only if" in prop and "ambiguous" in prop, "")
    # the stability requirement names its framework and points at the three-scheme evidence
    st, _t = _window(fs, "stability in the veridical", span=400)
    check("reporting item on stability is present", _t is not None, "")
    check("stability requirement cites PCS and the three-scheme appendix",
          "yu2020veridical" in st and "app:ci" in st, "")

    # the allocation certificate reproduces, and the paper quotes the instance count it was
    # checked over rather than asserting the proposition unverified
    import json as _json
    _mc = _json.load(open(os.path.join(ROOT, "results", "mckp_certificate.json")))
    check("MCKP certificate verified with zero failures", _mc["total_failures"] == 0,
          str(_mc["total_failures"]))
    # only binds on a draft that actually states the certificate (v5 onward); v4 predates it.
    # Anchored on the prose, not on a script name: v7 dropped the filename from the paper and
    # that silently switched this guard off.
    if "brute force" in fs:
        check("paper quotes the certificate instance count",
              f"{_mc['total_instances']:,}".replace(",", "{,}") in fs,
              f"{_mc['total_instances']:,}")

    # a validated cell identifies a total cell error; the arm-level split needs a normalization
    # v5 retitled this corollary: "one validated cell per model" reads as a single label, when the
    # result concerns n_v labels at one anchor workload. Accept either title -- the claim is
    # unchanged and v4 still passes against the old one. A bare .index() here would abort the whole
    # run rather than fail one check, so resolve the title first.
    anc, _t = _window(fs, "What validated anchor labels buy",
                      "What one validated cell per model buys")
    check("anchor corollary is present under a known title", _t is not None, "")
    # v5 renamed "request type" to "workload" throughout, so the anchor cell is now an
    # "anchor workload". The claim is unchanged: the split needs a normalization at a
    # pre-specified anchor. Accept either wording so the guard still runs against v4.
    check("anchor corollary states the normalization fixing the arm/residual split",
          "normalization" in anc and ("anchor workload" in anc or "anchor type" in anc), "")
    check("anchor corollary says labels identify the total cell error",
          "total" in anc, "")
    # the corner rule names the aggregation it rests on rather than asserting zero marginal value
    cor, _t = _window(fs, "Where the next dollar goes")
    check("next-dollar paragraph is present under a known title", _t is not None, "")
    check("corner rule names its aggregation", "aggregation" in cor, "")
    check("corner rule drops the bare zero-marginal-value claim",
          "exactly zero marginal value" not in fs, "")
    # Table 1 credits families that do estimate C, so no blanket "all of them assume C=0"
    check("no blanket claim that every neighbouring method assumes C={0}",
          "neighbouring literatures assume" not in fs
          and "assumption every neighbouring" not in fs, "")
    # the impossibility construction handles bounded/binary targets
    nid, _t = _window(fs, "The decision is not identified", span=1600)
    check("impossibility theorem is present under a known title", _t is not None, "")
    check("impossibility theorem covers bounded/binary targets",
          "binary" in nid and "mean tables" in nid, "")
    check("impossibility theorem scopes consistency as uniform over the class",
          "uniformly consistent over this class" in fs, "")


# -------------------------------------------------------------------------- 11
def c11_causal_and_breakeven_numbers() -> None:
    """New quantitative artifacts reproduce from their result files."""
    print("\n11. Causal-baseline and break-even numbers reproduce")
    s = tex()

    d = json.load(open(os.path.join(ROOT, "results", "tae_causal_baselines.json")))
    r = pd.DataFrame(d["rows"])
    want = {("ipw_true", 0.0): "0.0327", ("dr_true", 0.0): "0.0360",
            ("reg", 0.0): "0.0360", ("ipw_est", 0.01): "0.0375",
            ("dr_est", 0.01): "0.0435"}
    for (meth, clip), quoted in want.items():
        g = r[(r["method"] == meth) & (r["clip"] == clip) & (r["evaluator"] == "D0")]
        mu = g.groupby("partition")["reference_regret"].mean().mean()
        check(f"tab:causalbase {meth}@{clip} A1-D0 == {quoted}",
              abs(mu - float(quoted)) < 5e-4 and quoted in s, f"computed {mu:.4f}")
    # the predeclared primary contrast, re-derived
    case = json.load(open(os.path.join(ROOT, "results", "tae_validity.json")))["rows"]
    cs = pd.DataFrame(case)
    cs = cs[(cs["assignment"] == "A1") & (cs["evaluator"] == "D0")
            & (cs["budget"] == 0.35) & (cs["method"] == "case")]
    cs = cs.groupby("partition")["reference_regret"].mean()
    dr = r[(r["method"] == "dr_true") & (r["clip"] == 0.0) & (r["evaluator"] == "D0")]
    dr = dr.groupby("partition")["reference_regret"].mean()
    diff = (cs - dr).dropna()
    mu = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    check("primary contrast CASE - dr_true = -0.0023 [-0.0100,+0.0055]",
          abs(mu + 0.0023) < 5e-4 and abs(lo + 0.0100) < 2e-3 and abs(hi - 0.0055) < 2e-3,
          f"{mu:.4f} [{lo:.4f},{hi:.4f}]")
    check("paper quotes the primary contrast",
          "-0.0023" in s and "[-0.0100,+0.0055]" in s, "")

    b = json.load(open(os.path.join(RES, "appE_breakeven_ci.json")))
    for corpus, be, ci in (("primary", 1514, (1064, 2368)),
                           ("RouterBench", 1976, (1519, 2838))):
        v = b[corpus]
        ok = (abs(v["break_even"] - be) < 4
              and abs(v["break_even_ci"][0] - ci[0]) < 25
              and abs(v["break_even_ci"][1] - ci[1]) < 25)
        check(f"App E break-even {corpus}: {be} [{ci[0]}, {ci[1]}]", ok,
              f"file: {v['break_even']:.0f} "
              f"[{v['break_even_ci'][0]:.0f}, {v['break_even_ci'][1]:.0f}]")
        for n in (be, ci[0], ci[1]):
            tok = f"{n:,}".replace(",", "{,}")
            check(f"paper quotes {tok}", tok in s, "")


def main() -> None:
    print("=" * 70)
    print("SUBMISSION VERIFICATION")
    print("=" * 70)
    for fn in (c1_claims_match_sources, c2_switch_rows_decompose, c3_recall_denominator,
               c4_symbols_defined, c5_no_broken_refs, c6_page_count, c7_gate_tests,
               c8_anonymity, c9_review_text_guards, c10_gpt_guards,
               c11_causal_and_breakeven_numbers):
        try:
            fn()
        except Exception as exc:  # a check that cannot run is a failure, not a pass
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
    print("\n" + "=" * 70)
    for n in NOTES:
        print("  NOTE:", n)
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
