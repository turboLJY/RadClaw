#!/usr/bin/env python3
"""
Export a RadClaw trace as a single self-contained HTML file.

All images are embedded as base64 — no server needed, no images/ directory needed.
Just open the output file in any browser.

Usage:
    python scripts/export_report.py output/traces_demo/sample_13
    python scripts/export_report.py output/traces_demo/sample_13 --out ~/Desktop/sample13.html
"""

import argparse
import base64
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_report_html import (
    CSS,
    render_header,
    render_impressions,
    render_issues,
    render_all_retrieved_evidence,
)


# ---------------------------------------------------------------------------
# Self-contained versions of renderers (use inline base64 instead of paths)
# ---------------------------------------------------------------------------

def load_b64(img_id: str, images_dir: Path) -> str | None:
    p = images_dir / f"{img_id}.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None


def inline_img(b64: str, alt: str = "", style: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{escape(alt)}" style="{style}">'


def render_orig_and_report_inline(orig_b64: str | None, nl_report: str) -> str:
    img_html = ""
    if orig_b64:
        img_html = f"""
<div class="section">
  <div class="section-title">Original Radiograph</div>
  <div class="image-panel">
    {inline_img(orig_b64, "Original X-ray",
                "width:100%;border-radius:6px;cursor:pointer")}
    <div class="image-caption">Click to enlarge</div>
  </div>
</div>
"""
    report_html = f"""
<div class="section">
  <div class="section-title">Natural Language Report</div>
  <div class="report-box">{escape(nl_report or "(no report generated)")}</div>
</div>
"""
    if orig_b64:
        return f'<div class="main-layout">{img_html}{report_html}</div>'
    return report_html


def render_finding_img_ev_inline(ev_list: list, images_dir: Path) -> str:
    if not ev_list:
        return ""
    items = []
    for ev in ev_list:
        img_id = ev.get("image_id", "")
        desc = ev.get("description", img_id)
        caption = ev.get("finding_description", "") or desc
        b64 = load_b64(img_id, images_dir)
        if b64:
            items.append(f"""
<div class="img-thumb-wrap" onclick="openLightbox('{img_id}', `{escape(caption)}`)" title="{escape(caption)}">
  <img class="img-thumb" src="data:image/png;base64,{b64}" alt="{escape(img_id)}">
  <div class="img-thumb-label">{escape(img_id)}</div>
</div>""")
        else:
            tools = " → ".join(ev.get("tool_chain", []))
            items.append(f"""
<div class="img-thumb-wrap" title="{escape(desc)}\nTools: {escape(tools)}">
  <div class="no-image-placeholder">{escape(img_id)}<br><span style="font-size:0.6rem">(no image)</span></div>
  <div class="img-thumb-label">{escape(img_id)}</div>
</div>""")
    return f"""
<div class="evidence-section">
  <div class="evidence-label">Image evidence ({len(ev_list)})</div>
  <div class="img-evidence-row">{"".join(items)}</div>
</div>
"""


def render_finding_ret_ev(ev_list: list) -> str:
    if not ev_list:
        return ""
    from scripts.generate_report_html import chip
    items = []
    for ev in ev_list:
        eid = escape(ev.get("evidence_id", ""))
        src = ev.get("source_type", "")
        score = ev.get("relevance_score", 0.0)
        content = escape(ev.get("content", "")[:200])
        meta = ev.get("metadata", {})
        condition = escape(meta.get("condition", "") or meta.get("title", ""))
        items.append(f"""
<div class="ret-ev-item">
  <span class="ret-ev-id">{eid}</span>
  {chip(src)}
  <span class="ret-ev-score">{score:.3f}</span>
  {f'<div style="color:#7ec8e3;font-size:0.75rem;margin-top:2px">{condition}</div>' if condition else ""}
  <div class="ret-ev-content">{content}{"..." if len(ev.get("content","")) > 200 else ""}</div>
</div>""")
    return f"""
<div class="evidence-section" style="margin-top:12px">
  <div class="evidence-label">Retrieved evidence ({len(ev_list)})</div>
  <div class="ret-evidence-list">{"".join(items)}</div>
</div>
"""


def render_findings_inline(findings: list, images_dir: Path) -> str:
    if not findings:
        return ""
    cards = []
    for f in findings:
        fid = escape(f.get("finding_id", ""))
        ftype = f.get("type", "abnormal")
        anatomy = escape(f.get("anatomy", ""))
        obs = escape(f.get("observation", ""))
        text = escape(f.get("text", ""))
        conf = f.get("confidence", 0.0)
        support = f.get("support", {})
        img_ev = support.get("image_evidence", [])
        ret_ev = support.get("retrieved_evidence", [])
        type_cls = f"type-{ftype}" if ftype in ("normal", "abnormal", "uncertain") else "type-abnormal"
        obs_html = f'<span style="color:#94a3b8;font-size:0.8rem"> · {obs}</span>' if obs else ""
        cards.append(f"""
<div class="finding-card">
  <div class="finding-header">
    <span class="finding-id">{fid}</span>
    <span class="finding-type-badge {type_cls}">{ftype}</span>
    <span class="finding-anatomy">{anatomy}{obs_html}</span>
    <span class="finding-confidence">conf {conf:.2f}</span>
  </div>
  <div class="finding-body">
    <div class="finding-text">{text}</div>
    {render_finding_img_ev_inline(img_ev, images_dir)}
    {render_finding_ret_ev(ret_ev)}
  </div>
</div>""")
    return f"""
<div class="section">
  <div class="section-title">Findings ({len(findings)})</div>
  <div class="findings-grid">{"".join(cards)}</div>
</div>
"""


def render_all_img_ev_inline(img_ev_list: list, images_dir: Path) -> str:
    if not img_ev_list:
        return ""
    items = []
    for ev in img_ev_list:
        img_id = ev.get("image_id", "")
        desc = ev.get("description", img_id)
        finding_desc = ev.get("finding_description", "") or desc
        tools = " → ".join(ev.get("tool_chain", []))
        b64 = load_b64(img_id, images_dir)
        if b64:
            items.append(f"""
<div class="img-thumb-wrap" onclick="openLightbox('{img_id}', `{escape(finding_desc)}`)"
     title="{escape(desc)}\n{escape(tools)}">
  <img class="img-thumb" src="data:image/png;base64,{b64}" alt="{escape(img_id)}">
  <div class="img-thumb-label">{escape(img_id)}<br>
    <span style="color:#4b5563;font-size:0.6rem">{escape(tools[:30])}</span>
  </div>
</div>""")
        else:
            items.append(f"""
<div class="img-thumb-wrap" title="{escape(desc)}">
  <div class="no-image-placeholder">{escape(img_id)}</div>
  <div class="img-thumb-label">{escape(img_id)}</div>
</div>""")
    return f"""
<div class="section">
  <div class="section-title">All Image Evidence ({len(img_ev_list)})</div>
  <div class="image-panel">
    <div class="img-evidence-row" style="flex-wrap:wrap">{"".join(items)}</div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Lightbox — uses a JS map keyed by image_id, populated once at load time
# ---------------------------------------------------------------------------

def build_lightbox_js(img_ev_list: list, images_dir: Path, orig_b64: str | None) -> str:
    """Build a JS map of {image_id: base64_data_url} — each image embedded once."""
    entries = []
    seen = set()

    if orig_b64:
        entries.append(f'  "__orig__": "data:image/png;base64,{orig_b64}"')
        seen.add("__orig__")

    for ev in img_ev_list:
        img_id = ev.get("image_id", "")
        if img_id in seen:
            continue
        b64 = load_b64(img_id, images_dir)
        if b64:
            entries.append(f'  "{img_id}": "data:image/png;base64,{b64}"')
            seen.add(img_id)

    map_js = "{\n" + ",\n".join(entries) + "\n}"
    return f"""
const IMG_MAP = {map_js};

function openLightbox(imgId, caption) {{
    const src = IMG_MAP[imgId];
    if (!src) return;
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox-caption').textContent = caption;
    document.getElementById('lightbox').classList.add('open');
}}
document.getElementById('lightbox').addEventListener('click', function() {{
    this.classList.remove('open');
    document.getElementById('lightbox-img').src = '';
}});
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_self_contained_html(trace_dir: Path) -> str:
    result_path = trace_dir / "result.json"
    if not result_path.exists():
        sys.exit(f"ERROR: {result_path} not found")

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    summary_path = trace_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    images_dir = trace_dir / "images"

    structured_report = result.get("structured_report") or {}
    findings = structured_report.get("findings", [])
    impressions = structured_report.get("impressions", [])
    nl_report = result.get("natural_language_report", "")
    img_ev_all = result.get("image_evidence", [])
    ret_ev_all = result.get("retrieved_evidence", [])
    remaining_issues = result.get("remaining_issues", [])

    orig_b64 = None
    if images_dir.exists():
        for p in sorted(images_dir.glob("IMG_*_001.png")):
            orig_b64 = base64.b64encode(p.read_bytes()).decode()
            break

    lightbox_js = build_lightbox_js(img_ev_all, images_dir, orig_b64)

    orig_onclick = f'onclick="openLightbox(\'__orig__\', \'Original Radiograph\')"' if orig_b64 else ""
    orig_b64_for_render = orig_b64  # render_orig_and_report_inline uses it

    # Patch: add onclick to original image
    orig_report_html = render_orig_and_report_inline(orig_b64_for_render, nl_report)
    if orig_b64:
        orig_report_html = orig_report_html.replace(
            "cursor:pointer\")",
            f'cursor:pointer\" {orig_onclick})',
        )

    agent_log_html = ""
    for e in result.get("agent_run_log", []):
        agent_log_html += (
            f"<tr><td>{escape(e.get('agent',''))}</td>"
            f"<td>{e.get('iteration','')}</td>"
            f"<td>{e.get('output_length','')}</td></tr>"
        )
    if agent_log_html:
        agent_log_html = f"""
<div class="section">
  <div class="section-title">Agent Run Log</div>
  <table class="ev-table">
    <thead><tr><th>Agent</th><th>Iteration</th><th>Output length (chars)</th></tr></thead>
    <tbody>{agent_log_html}</tbody>
  </table>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadClaw Report — {escape(result.get("session_id", ""))}</title>
<style>{CSS}</style>
</head>
<body>

<div id="lightbox">
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-caption"></div>
</div>

<div class="container">
  {render_header(result, summary or None)}
  {orig_report_html}
  {render_impressions(impressions)}
  {render_findings_inline(findings, images_dir)}
  {render_all_img_ev_inline(img_ev_all, images_dir)}
  {render_all_retrieved_evidence(ret_ev_all)}
  {render_issues(remaining_issues)}
  {agent_log_html}
</div>

<script>{lightbox_js}</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Export RadClaw trace as self-contained HTML (images embedded inline)"
    )
    parser.add_argument("trace_dir", help="Trace directory (e.g. output/traces_demo/sample_13)")
    parser.add_argument("--out", help="Output file path (default: <trace_dir>/report_standalone.html)")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.is_dir():
        sys.exit(f"ERROR: {trace_dir} is not a directory")

    out_path = Path(args.out).resolve() if args.out else trace_dir / "report_standalone.html"

    print(f"Reading:  {trace_dir}")
    html = generate_self_contained_html(trace_dir)
    out_path.write_text(html, encoding="utf-8")
    size_kb = len(html.encode()) / 1024
    print(f"Written:  {out_path}")
    print(f"Size:     {size_kb:.0f} KB")
    print(f"\nDownload and open locally:")
    print(f"  scp <server>:{out_path} ~/Desktop/")


if __name__ == "__main__":
    main()
