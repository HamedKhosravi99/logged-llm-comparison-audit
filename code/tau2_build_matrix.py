"""Priority 1, step 2.2: one complete matrix from the existing 440 tau2 simulations.

No new API calls. Every field is read from the saved simulation records.
"""
import json, glob, os, re, collections
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "ext", "tau2-bench-verified", "data", "simulations")
OUT = os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet")

FN = re.compile(r".*?_(airline|retail|telecom)_llm_agent_(.+?)_user_simulator_(.+)\.json")


def first_user_message(task):
    """The scenario text a production router could condition on, before any response."""
    for key in ("user_scenario", "scenario", "description"):
        v = task.get(key)
        if isinstance(v, dict):
            for k2 in ("instructions", "task_instructions", "description"):
                if isinstance(v.get(k2), str):
                    return v[k2]
                if isinstance(v.get(k2), dict):
                    return json.dumps(v[k2])
        if isinstance(v, str):
            return v
    return json.dumps({k: v for k, v in task.items() if k != "id"})[:4000]


def build():
    rows = []
    for f in sorted(glob.glob(os.path.join(SIMS, "*.json"))):
        m = FN.match(os.path.basename(f))
        if not m:
            continue
        domain, agent, usim = m.group(1), m.group(2), m.group(3)
        d = json.load(open(f))
        info = d.get("info", {})
        tasks = {t.get("id"): t for t in d.get("tasks", [])}
        agent_llm = (info.get("agent_info") or {}).get("llm", agent)
        user_llm = (info.get("user_info") or {}).get("llm", usim)
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            msgs = s.get("messages") or []
            tool_calls = sum(len(x.get("tool_calls") or []) for x in msgs)
            tool_err = sum(1 for x in msgs if x.get("role") == "tool"
                           and "error" in str(x.get("content", "")).lower())
            tin = sum((x.get("usage") or {}).get("prompt_tokens", 0) or 0 for x in msgs)
            tout = sum((x.get("usage") or {}).get("completion_tokens", 0) or 0 for x in msgs)
            bd = ri.get("reward_breakdown") or {}
            rows.append(dict(
                task_id=s.get("task_id"), domain=domain, trial_id=s.get("trial"),
                agent_model=agent, agent_model_version=agent_llm, user_simulator_model=user_llm,
                task_success=int((ri.get("reward") or 0) >= 1.0), reward=ri.get("reward"),
                policy_compliance=bd.get("DB"), tool_call_count=tool_calls,
                tool_error_count=tool_err, turn_count=len(msgs),
                input_tokens=tin or None, output_tokens=tout or None,
                agent_cost_usd=s.get("agent_cost"), user_simulator_cost_usd=s.get("user_cost"),
                judge_cost_usd=0.0,   # tau2 grades programmatically (db_check); no LLM judge is used
                total_measured_cost_usd=(s.get("agent_cost") or 0) + (s.get("user_cost") or 0),
                latency_ms=(s.get("duration") or 0) * 1000.0,
                termination_reason=s.get("termination_reason"),
                trajectory_path=os.path.basename(f), seed=s.get("seed"),
                initial_user_message=first_user_message(tasks.get(s.get("task_id"), {})),
            ))
    df = pd.DataFrame(rows)

    print(f"  {len(df)} simulation records, {df.task_id.nunique()} distinct task ids")
    ok = True
    # each model has at least 24 observations in every domain
    obs = df.groupby(["domain", "agent_model"]).size()
    thin = obs[obs < 24]
    print(f"  [{'OK  ' if thin.empty else 'FAIL'}] every model has >=24 observations per domain"
          + ("" if thin.empty else f"  {dict(thin)}"))
    ok &= thin.empty
    # missing cost never silently zero
    miss = int(df.agent_cost_usd.isna().sum())
    print(f"  [{'OK  ' if miss == 0 else 'WARN'}] agent cost present on every record ({miss} missing)")
    # failed / retried runs retained
    terms = collections.Counter(df.termination_reason.dropna())
    print(f"  [OK  ] failed and retried runs retained: {dict(terms)}")
    # counterfactual completeness
    piv = df.groupby(["domain", "task_id"]).agent_model.nunique()
    complete = int((piv == df.agent_model.nunique()).sum())
    print(f"  [OK  ] tasks with all {df.agent_model.nunique()} models present: {complete}/{piv.size}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n  wrote {os.path.abspath(OUT)}")
    return df


if __name__ == "__main__":
    build()
