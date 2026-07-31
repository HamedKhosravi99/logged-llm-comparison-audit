"""Gradio demonstration of the offline RouterAudit package.

This file is an interface only. Every number shown is computed inside `routeraudit`, which is
covered by the test suite. Nothing here calls a model API or changes production routing.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from routeraudit import AuditConfig, SchemaError, __version__, analyze_trace

HERE = Path(__file__).parent
EXAMPLES = HERE / "examples"
TEMPLATES = HERE / "templates"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

PRIVACY = (
    "**Upload only synthetic, anonymized, or non-sensitive data.** Do not upload confidential "
    "conversations, personal information, credentials, or proprietary prompts to this public "
    "demonstration. Run RouterAudit locally for confidential traces."
)


def read_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if p.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB demonstration limit.")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError("Only CSV and Parquet files are accepted.")


def write_bundle(result, out: Path) -> str:
    result.summary.to_csv(out / "summary.csv", index=False)
    result.audit_queue.to_csv(out / "audit_queue.csv", index=False)
    result.candidate_switches.to_csv(out / "candidate_switches.csv", index=False)
    (out / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, default=str), encoding="utf-8")
    (out / "assumptions.json").write_text(
        json.dumps(result.manifest, indent=2, default=str), encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, default=str), encoding="utf-8")
    (out / "warnings.json").write_text(
        json.dumps(result.warnings, indent=2), encoding="utf-8")
    (out / "report.html").write_text(result.report_html, encoding="utf-8")
    zip_path = out / "routeraudit_report.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f != zip_path:
                z.write(f, arcname=f.name)
    return str(zip_path)


def load_example():
    """Populate the upload fields from the bundled synthetic example."""
    return (str(EXAMPLES / "routed_log.csv"), str(EXAMPLES / "model_prices.csv"),
            str(EXAMPLES / "randomized_audit.csv"), str(EXAMPLES / "verified_outcomes.csv"),
            str(EXAMPLES / "traffic.csv"))


def run_analysis(routed_log_file, prices_file, audit_file, verified_file, traffic_file,
                 tolerance, future_requests, audit_budget, verification_cost,
                 confidence, minimum_support, readout_size):
    empty = pd.DataFrame()
    try:
        routed_log = read_table(routed_log_file)
        prices = read_table(prices_file)
        if routed_log is None or prices is None:
            raise ValueError("A routed log and a model-price table are required. "
                             "Use 'Load bundled example' on the Data tab to try the app.")

        config = AuditConfig(
            quality_tolerance=float(tolerance),
            future_requests=int(future_requests),
            audit_budget_usd=float(audit_budget),
            verification_cost_usd=float(verification_cost),
            confidence_level=float(confidence),
            minimum_support=int(minimum_support),
            audit_readout_size=int(readout_size),
        )
        result = analyze_trace(
            routed_log=routed_log, model_prices=prices, config=config,
            randomized_audit=read_table(audit_file),
            verified_outcomes=read_table(verified_file),
            traffic=read_table(traffic_file),
        )

        run_dir = Path(tempfile.mkdtemp(prefix=f"routeraudit-{uuid.uuid4().hex[:8]}-"))
        zip_path = write_bundle(result, run_dir)

        mode = ("**Mode A — audit plan.** No randomized audit was supplied, so this run proposes "
                "what evidence to collect. It does not claim any switch is valid."
                if result.mode == "plan" else
                "**Mode B — completed-audit analysis.** Estimates are routing-adjusted and pooled "
                "by inverse variance.")
        status = f"### Analysis completed\n\n{mode}\n"
        if result.warnings:
            status += "\n**Warnings**\n\n" + "\n".join(f"- {w}" for w in result.warnings)
        return (status, result.summary, result.audit_queue, result.candidate_switches,
                result.diagnostics, gr.DownloadButton(value=zip_path, visible=True))

    except (SchemaError, ValueError) as exc:
        return (f"### Analysis failed\n\n**{type(exc).__name__}:** {exc}",
                empty, empty, empty, {}, gr.DownloadButton(visible=False))
    except Exception as exc:  # noqa: BLE001 - surface the class, never the uploaded rows
        return (f"### Analysis failed\n\n**{type(exc).__name__}:** {exc}",
                empty, empty, empty, {}, gr.DownloadButton(visible=False))


with gr.Blocks(title="RouterAudit", analytics_enabled=False) as demo:
    gr.Markdown(f"""
# RouterAudit

Audit planning and uncertainty reporting for quality-constrained model-switch decisions.

This demonstration **does not call models, modify production routing, guarantee savings, or
establish measurement validity unless verified outcomes are supplied.** Package version
`{__version__}`.
""")

    with gr.Tab("1. Data"):
        gr.Markdown(PRIVACY)
        example_btn = gr.Button("Load bundled example", variant="secondary")
        with gr.Row():
            routed_log_file = gr.File(label="Routed log (required)",
                                      file_types=[".csv", ".parquet"], type="filepath")
            prices_file = gr.File(label="Model prices (required)",
                                  file_types=[".csv", ".parquet"], type="filepath")
        with gr.Row():
            audit_file = gr.File(label="Randomized audit results (optional)",
                                 file_types=[".csv", ".parquet"], type="filepath")
            verified_file = gr.File(label="Verified target outcomes (optional)",
                                    file_types=[".csv", ".parquet"], type="filepath")
        traffic_file = gr.File(label="Traffic weights (optional)",
                               file_types=[".csv", ".parquet"], type="filepath")
        gr.Markdown("**Input templates** — download, fill in, and upload.")
        gr.File(label="Templates",
                value=[str(p) for p in sorted(TEMPLATES.glob("*.csv"))],
                interactive=False)

    with gr.Tab("2. Decision settings"):
        gr.Markdown("Every default below is recorded in `assumptions.json` inside the report.")
        tolerance = gr.Number(label="Quality tolerance (delta)", value=0.02, minimum=0)
        future_requests = gr.Number(label="Expected future requests", value=100_000,
                                    minimum=1, precision=0)
        audit_budget = gr.Number(label="Maximum audit budget ($)", value=10.0, minimum=0)
        verification_cost = gr.Number(label="Verification cost per validated outcome ($)",
                                      value=0.0, minimum=0)
        confidence = gr.Slider(label="Confidence level", minimum=0.5, maximum=0.999,
                               value=0.95, step=0.001)
        minimum_support = gr.Number(label="Minimum logged rows per model/request-type pair",
                                    value=8, minimum=1, precision=0)
        readout_size = gr.Number(label="Randomized readouts per audited pair", value=30,
                                 minimum=1, precision=0)
        run_button = gr.Button("Run RouterAudit", variant="primary")

    with gr.Tab("3. Results"):
        status = gr.Markdown()
        summary = gr.Dataframe(label="Decision summary", interactive=False, wrap=True)
        audit_queue = gr.Dataframe(label="Proposed audit queue", interactive=False, wrap=True)
        switches = gr.Dataframe(label="Candidate switches", interactive=False, wrap=True)
        diagnostics = gr.JSON(label="Diagnostics and assumptions")

    with gr.Tab("4. Report"):
        gr.Markdown(
            "The ZIP contains `summary.csv`, `audit_queue.csv`, `candidate_switches.csv`, "
            "`diagnostics.json`, `assumptions.json`, `run_manifest.json`, `warnings.json` "
            "and `report.html`.")
        report_download = gr.DownloadButton(label="Download complete report", visible=False)

    example_btn.click(fn=load_example, inputs=None,
                      outputs=[routed_log_file, prices_file, audit_file, verified_file,
                               traffic_file])
    run_button.click(
        fn=run_analysis,
        inputs=[routed_log_file, prices_file, audit_file, verified_file, traffic_file,
                tolerance, future_requests, audit_budget, verification_cost,
                confidence, minimum_support, readout_size],
        outputs=[status, summary, audit_queue, switches, diagnostics, report_download],
    )

if __name__ == "__main__":
    demo.queue().launch()
