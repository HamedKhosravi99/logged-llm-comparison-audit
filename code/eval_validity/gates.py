"""Held-out decision checks (conditional deployment gates) for the audited candidate policy.

The deployment question is not "is the audited surface accurate?" but "should we act on the audit
instead of on the log?". The status quo is therefore the LOG-DERIVED DEPLOYMENT POLICY -- the
model-selection conclusion the operational log supports today -- and not the incumbent router that
generated the log (that variant is `kind="router"` below, and is reported only in the appendix).
A held-out decision check buys a few randomized readouts on the cells the two policies disagree
about, on a cohort of prompts no other stage touched, and switches only if the candidate wins.

Which evaluator reads those verification prompts is the whole point of the experiment:

  same_evaluator_decision_check   the verification readouts come from the operational evaluator
                                  (stream v, distortion C^v). This is what a production team can
                                  actually run, and it inherits the evaluator's construct bias.
  reference_channel_decision_check the verification readouts come from the designated target
                                  channel. This is an explicitly labelled lower-distortion control,
                                  NOT a claim that such a channel is available in production.

These are conditional deployment gates, not safe-deployment certificates: they bound harm relative
to the log-derived baseline on the cells they check, under the evaluator that checks them.
"""
import numpy as np


def baseline_policy(base, surf, supp, kind="logged"):
    """The status quo the audited candidate would replace.

    kind="logged" (default, main paper): the log-derived deployment policy
        pi_base(j) = argmax_k M_kj -- what an evaluation pipeline would ship if it believed its own
        logged comparison. The gate then arbitrates "does held-out evidence support acting on the
        audit rather than on the log?".
    kind="router": the incumbent router's own per-context choice, i.e. the policy that actually
        generated the log. Appendix variant only: that baseline is much weaker, so switching to the
        audited candidate is almost never harmful and the comparison is uninformative.
    """
    J = base["J"]
    M = surf["M"]
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if not len(ks):
            continue
        if kind == "router" and supp[base["inc"][j], j]:
            pol[j] = int(base["inc"][j])
        else:
            pol[j] = int(ks[int(np.argmax(M[ks, j]))])
    return pol


def _read(pool, rng, n_read):
    return float(pool[rng.integers(0, len(pool), size=min(n_read, len(pool)))].mean())


def ungated_candidate(pol_cand, **_):
    """No verification: ship the audited candidate."""
    return dict(deploy_candidate=True, n_ver_cells=0, v_hat_cand=None, v_hat_base=None)


def held_out_decision_check(pol_base, pol_cand, verpool, supp, mv, seed, n_read=30):
    """Buy up to `mv` verification readouts on the cells the two policies disagree about.

    Readouts are bought in matched pairs (candidate cell and baseline cell of the same context),
    so the held-out comparison is paired. Contexts where the policies already agree carry no
    decision risk and are only used if budget remains.
    """
    rng = np.random.default_rng(seed + 777)
    J = len(pol_base)
    diff = [j for j in range(J) if pol_base[j] >= 0 and pol_cand[j] >= 0 and pol_base[j] != pol_cand[j]]
    rng.shuffle(diff)
    reads, used = {}, 0
    for j in diff:
        if used + 2 > mv:
            break
        for k in (pol_cand[j], pol_base[j]):
            reads[(int(k), j)] = _read(verpool[(int(k), j)], rng, n_read)
        used += 2
    paired = [j for j in diff if (int(pol_cand[j]), j) in reads and (int(pol_base[j]), j) in reads]
    if not paired:
        # nothing to check (policies agree, or no budget for a pair): the switch is a no-op or
        # unverified; default to the audited candidate, exactly as the ungated arm would.
        return dict(deploy_candidate=True, n_ver_cells=used, v_hat_cand=None, v_hat_base=None,
                    n_contested=len(diff), n_checked=0, verdicts=[])
    v_cand = float(np.mean([reads[(int(pol_cand[j]), j)] for j in paired]))
    v_base = float(np.mean([reads[(int(pol_base[j]), j)] for j in paired]))
    verdicts = [(j, float(reads[(int(pol_cand[j]), j)] - reads[(int(pol_base[j]), j)]))
                for j in paired]
    return dict(deploy_candidate=bool(v_cand >= v_base), n_ver_cells=used,
                v_hat_cand=v_cand, v_hat_base=v_base, n_contested=len(diff), n_checked=len(paired),
                verdicts=verdicts, contested=diff)


def channel_verdict_validity(pol_base, pol_cand, surf, contested):
    """Does the checking channel give the RIGHT per-context verdict, with unlimited budget?

    For every contested context this compares the sign of the channel's population mean difference
    (candidate cell minus baseline cell, over the whole verification cohort) with the sign of the
    same difference measured in the designated target channel on the same prompts. The reference
    channel agrees by construction; the operational channel need not. Sampling noise is removed, so
    what remains is the systematic verdict error a bigger verification budget cannot fix.
    """
    out = {}
    for name, pool in (("same_evaluator", surf["verq"]), ("reference_channel", surf["verq_ref"])):
        agree, flip = 0, 0
        for j in contested:
            kc, ki = int(pol_cand[j]), int(pol_base[j])
            d_ch = float(pool[(kc, j)].mean() - pool[(ki, j)].mean())
            d_rf = float(surf["verq_ref"][(kc, j)].mean() - surf["verq_ref"][(ki, j)].mean())
            if abs(d_rf) < 1e-12:
                continue
            agree += int(np.sign(d_ch) == np.sign(d_rf))
            flip += int(np.sign(d_ch) != np.sign(d_rf))
        n = agree + flip
        out[name] = float(flip / n) if n else float("nan")
        out[name + "_n"] = n
    return out


def same_evaluator_decision_check(pol_base, pol_cand, surf, supp, mv, seed, n_read=30):
    """Verification read by the SAME operational evaluator that produced the log."""
    return held_out_decision_check(pol_base, pol_cand, surf["verq"], supp, mv, seed, n_read)


def reference_channel_decision_check(pol_base, pol_cand, surf, supp, mv, seed, n_read=30):
    """Verification read by the designated target channel (labelled lower-distortion control)."""
    return held_out_decision_check(pol_base, pol_cand, surf["verq_ref"], supp, mv, seed, n_read)
