"""Self-contained HTML report. No external assets, no network references."""
from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
from jinja2 import Template

_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RouterAudit report</title>
<style>
 body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
      margin:2rem auto;max-width:60rem;padding:0 1rem;color:#1a1a1a;background:#fff}
 h1{font-size:1.5rem;margin-bottom:.2rem} h2{font-size:1.1rem;margin-top:2rem}
 .sub{color:#555;margin-top:0}
 table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:13px}
 th,td{border-bottom:1px solid #e3e3e3;padding:.35rem .5rem;text-align:left;vertical-align:top}
 th{background:#f6f7f9;font-weight:600}
 td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
 .warn{background:#fff8e6;border-left:3px solid #e8a33d;padding:.6rem .8rem;margin:.4rem 0}
 .note{background:#f2f5fa;border-left:3px solid #2a78d6;padding:.6rem .8rem}
 pre{background:#f6f7f9;padding:.7rem;overflow-x:auto;font-size:12px;border-radius:3px}
 .empty{color:#666;font-style:italic}
</style></head><body>
<h1>RouterAudit report</h1>
<p class="sub">Mode: <strong>{{ mode }}</strong> &middot; generated {{ manifest.timestamp_utc }}
 &middot; routeraudit {{ manifest.routeraudit_version }}</p>

<div class="note">This report plans and analyses an audit of a routed log. It does not call
models, does not modify routing, and does not guarantee savings.
{% if not manifest.measurement_validity_established %}
No verified target outcomes were supplied, so <strong>measurement validity is not
established</strong>.{% endif %}</div>

<h2>1. Decision summary</h2>
{{ summary }}

<h2>2. Warnings</h2>
{% if warnings %}{% for w in warnings %}<div class="warn">{{ w }}</div>{% endfor %}
{% else %}<p class="empty">None.</p>{% endif %}

<h2>3. Proposed audit queue</h2>
{{ queue }}

<h2>4. Candidate switches</h2>
{{ switches }}

<h2>5. Diagnostics</h2>
<pre>{{ diagnostics }}</pre>

<h2>6. Assumptions and run manifest</h2>
<pre>{{ manifest_json }}</pre>
</body></html>""")


def _table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return '<p class="empty">No rows.</p>'
    num = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    head = "".join(
        f'<th class="num">{html.escape(str(c))}</th>' if c in num
        else f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, r):
            if c in num and isinstance(v, float):
                cells.append(f'<td class="num">{v:,.6g}</td>')
            elif c in num:
                cells.append(f'<td class="num">{html.escape(str(v))}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_report(mode: str, summary: pd.DataFrame, queue: pd.DataFrame,
                  switches: pd.DataFrame, diagnostics: dict[str, Any],
                  warnings: list[str], manifest: dict[str, Any]) -> str:
    return _TEMPLATE.render(
        mode=mode,
        summary=_table(summary),
        queue=_table(queue),
        switches=_table(switches),
        diagnostics=html.escape(json.dumps(diagnostics, indent=2, default=str)),
        manifest_json=html.escape(json.dumps(manifest, indent=2, default=str)),
        manifest=manifest,
        warnings=[html.escape(w) for w in warnings],
    )
