#!/usr/bin/env python3
"""
Generate a standalone-layout HTML report from a RadClaw trace directory.

Usage:
    python scripts/generate_report_html.py output/traces/sample_12
    python scripts/generate_report_html.py output/traces/sample_12 --out report.html

The HTML references images from a sibling images/ directory.
Keep report.html alongside the images/ folder when moving.
"""

import argparse
import json
import sys
from html import escape
from pathlib import Path


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    line-height: 1.6;
}
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

/* Header */
.header {
    background: linear-gradient(135deg, #1a1f2e 0%, #252d3e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 24px;
}
.header h1 { font-size: 1.8rem; color: #7ec8e3; margin-bottom: 8px; }
.header-meta { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; }
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-pass  { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-fail  { background: #3a1a1a; color: #f87171; border: 1px solid #991b1b; }
.badge-info  { background: #1a2a3a; color: #93c5fd; border: 1px solid #1e3a5f; }
.badge-warn  { background: #2a2a1a; color: #fbbf24; border: 1px solid #92400e; }

/* Section titles */
.section { margin-bottom: 28px; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1px;
    padding-bottom: 8px; border-bottom: 1px solid #2d3748;
    margin-bottom: 16px;
}

/* Two-column layout */
.main-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 900px) { .main-layout { grid-template-columns: 1fr; } }

/* Image panels */
.image-panel {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
}
.image-caption {
    font-size: 0.78rem; color: #64748b;
    text-align: center; margin-top: 8px;
}

/* Natural language report */
.report-box {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 20px;
    font-size: 0.9rem;
    white-space: pre-wrap;
    color: #cbd5e1;
    overflow-y: auto;
    max-height: 540px;
}

/* Findings */
.findings-grid { display: flex; flex-direction: column; gap: 16px; }
.finding-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    overflow: hidden;
}
.finding-header {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px;
    background: #1e2535;
    border-bottom: 1px solid #2d3748;
}
.finding-id { font-weight: 700; color: #7ec8e3; font-size: 0.95rem; }
.finding-type-badge {
    font-size: 0.72rem; font-weight: 700;
    padding: 2px 8px; border-radius: 10px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.type-normal   { background: #1a3a2a; color: #4ade80; }
.type-abnormal { background: #3a1a1a; color: #f87171; }
.type-uncertain { background: #2a2a1a; color: #fbbf24; }
.finding-anatomy { color: #94a3b8; font-size: 0.85rem; }
.finding-confidence { margin-left: auto; font-size: 0.78rem; color: #64748b; }

.finding-body { padding: 14px 16px; }
.finding-text { color: #e2e8f0; font-size: 0.9rem; margin-bottom: 12px; }

/* Evidence in findings */
.evidence-section { margin-top: 10px; }
.evidence-label {
    font-size: 0.72rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.img-evidence-row { display: flex; gap: 10px; flex-wrap: wrap; }
.img-thumb-wrap { position: relative; cursor: pointer; }
.img-thumb {
    width: 100px; height: 100px; object-fit: cover;
    border-radius: 6px; border: 1px solid #374151;
    transition: border-color 0.2s; display: block;
}
.img-thumb:hover { border-color: #7ec8e3; }
.img-thumb-label {
    font-size: 0.65rem; color: #94a3b8;
    text-align: center; margin-top: 4px;
    max-width: 100px; word-wrap: break-word;
}
.no-image-placeholder {
    width: 100px; height: 100px; border-radius: 6px;
    border: 1px dashed #374151;
    display: flex; align-items: center; justify-content: center;
    color: #4b5563; font-size: 0.65rem; text-align: center;
}

.ret-evidence-list { display: flex; flex-direction: column; gap: 6px; }
.ret-ev-item {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.78rem;
}
.ret-ev-id { color: #7ec8e3; font-weight: 600; margin-right: 6px; }
.ret-ev-score { color: #94a3b8; font-size: 0.7rem; }
.ret-ev-content { color: #9ca3af; margin-top: 4px; font-size: 0.75rem; }

/* Impressions */
.impression-card {
    background: #1a1f2e;
    border: 1px solid #3b4a5a;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.impression-id { font-weight: 700; color: #93c5fd; margin-bottom: 6px; }
.impression-text { color: #e2e8f0; font-size: 0.9rem; }
.impression-links { margin-top: 8px; font-size: 0.75rem; color: #64748b; }

/* All evidence table */
.ev-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.ev-table th {
    text-align: left; padding: 8px 12px;
    background: #1e2535; color: #94a3b8;
    border-bottom: 1px solid #2d3748;
    font-weight: 600; text-transform: uppercase; font-size: 0.72rem;
    letter-spacing: 0.5px;
}
.ev-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1f2937;
    vertical-align: top;
}
.ev-table tr:hover td { background: #1a2030; }
.score-bar-wrap { display: flex; align-items: center; gap: 6px; }
.score-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #3b82f6, #7ec8e3);
    min-width: 2px;
}
.source-chip {
    display: inline-block; padding: 1px 7px; border-radius: 8px;
    font-size: 0.68rem; font-weight: 600;
}
.src-guideline  { background: #1a3050; color: #93c5fd; }
.src-pubmed     { background: #1a3a2a; color: #4ade80; }
.src-concept    { background: #2a1a3a; color: #c084fc; }
.src-case       { background: #2a2a1a; color: #fbbf24; }
.src-other      { background: #2a2020; color: #f87171; }

/* Lightbox */
#lightbox {
    display: none; position: fixed; top: 0; left: 0;
    width: 100%; height: 100%; background: rgba(0,0,0,0.92);
    z-index: 1000; align-items: center; justify-content: center;
    flex-direction: column; gap: 16px;
}
#lightbox.open { display: flex; }
#lightbox-img {
    max-width: 90vw; max-height: 82vh;
    border-radius: 8px; border: 1px solid #374151; object-fit: contain;
}
#lightbox-caption {
    color: #94a3b8; font-size: 0.85rem; background: rgba(0,0,0,0.7);
    padding: 6px 20px; border-radius: 20px; max-width: 80vw; text-align: center;
}

/* Issues */
.issue-card {
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
    font-size: 0.83rem;
}
.sev-low      { background: #1a2a1a; border-left: 3px solid #4ade80; }
.sev-medium   { background: #2a2a1a; border-left: 3px solid #fbbf24; }
.sev-high     { background: #2a1a1a; border-left: 3px solid #f97316; }
.sev-critical { background: #3a1a1a; border-left: 3px solid #f87171; }
.issue-top { display: flex; gap: 10px; align-items: center; margin-bottom: 4px; }
.issue-id { font-weight: 700; color: #94a3b8; }
.issue-type { font-size: 0.72rem; color: #64748b; }
.issue-attr { font-size: 0.72rem; background: #1e2535; color: #93c5fd; padding: 1px 7px; border-radius: 8px; }
.issue-desc { color: #e2e8f0; }
.issue-fix { color: #94a3b8; margin-top: 4px; font-size: 0.78rem; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOURCE_CHIP_CLASS = {
    "guideline": "src-guideline",
    "pubmed": "src-pubmed",
    "concept": "src-concept",
    "umls": "src-concept",
    "case": "src-case",
    "literature": "src-pubmed",
}


def chip(source_type: str) -> str:
    cls = SOURCE_CHIP_CLASS.get(source_type, "src-other")
    return f'<span class="source-chip {cls}">{escape(source_type)}</span>'


def score_bar(score: float) -> str:
    pct = max(2, int(score * 120))
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar" style="width:{pct}px"></div>'
        f'<span style="font-size:0.72rem;color:#94a3b8">{score:.3f}</span>'
        f'</div>'
    )


def img_path(img_id: str, images_dir: Path) -> str | None:
    p = images_dir / f"{img_id}.png"
    return f"images/{img_id}.png" if p.exists() else None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header(result: dict, summary: dict | None) -> str:
    sid = escape(result.get("session_id", "unknown"))
    iters = result.get("iterations", "?")
    passed = result.get("review_passed", False)
    img_cnt = result.get("image_evidence_count", 0)
    ret_cnt = result.get("retrieved_evidence_count", 0)
    elapsed = summary.get("elapsed_seconds", "?") if summary else "?"

    pass_badge = (
        '<span class="badge badge-pass">&#10003; Review Passed</span>' if passed
        else '<span class="badge badge-fail">&#10007; Review Not Passed</span>'
    )

    return f"""
<div class="header">
  <h1>RadClaw Radiology Report</h1>
  <div style="font-size:0.85rem;color:#64748b;margin-top:4px">Session: {sid}</div>
  <div class="header-meta">
    {pass_badge}
    <span class="badge badge-info">&#128260; {iters} iteration{'s' if iters != 1 else ''}</span>
    <span class="badge badge-info">&#128247; {img_cnt} image evidence</span>
    <span class="badge badge-info">&#128196; {ret_cnt} retrieved evidence</span>
    {'<span class="badge badge-warn">&#9201; ' + str(elapsed) + 's</span>' if elapsed != "?" else ""}
  </div>
</div>
"""


def render_original_and_report(orig_path: str | None, nl_report: str) -> str:
    img_html = ""
    if orig_path:
        img_html = f"""
<div class="section">
  <div class="section-title">Original Radiograph</div>
  <div class="image-panel">
    <img style="width:100%;border-radius:6px;cursor:pointer"
         src="{orig_path}" alt="Original X-ray"
         onclick="openLightbox(this.src, 'Original Radiograph')">
    <div class="image-caption">Click to enlarge</div>
  </div>
</div>
"""
    report_html = f"""
<div class="section">
  <div class="section-title">Natural Language Report</div>
  <div class="report-box">{escape(nl_report or '(no report generated)')}</div>
</div>
"""
    if orig_path:
        return f'<div class="main-layout">{img_html}{report_html}</div>'
    return report_html


def render_finding_image_evidence(ev_list: list, images_dir: Path) -> str:
    if not ev_list:
        return ""
    items = []
    for ev in ev_list:
        img_id = ev.get("image_id", "")
        desc = ev.get("description", img_id)
        caption = ev.get("finding_description", "") or desc
        p = img_path(img_id, images_dir)
        if p:
            items.append(f"""
<div class="img-thumb-wrap" onclick="openLightbox('images/{img_id}.png', `{escape(caption)}`)" title="{escape(caption)}">
  <img class="img-thumb" src="{p}" alt="{escape(img_id)}" loading="lazy">
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
  <div class="img-evidence-row">{''.join(items)}</div>
</div>
"""


def render_finding_ret_evidence(ev_list: list) -> str:
    if not ev_list:
        return ""
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
  {f'<div style="color:#7ec8e3;font-size:0.75rem;margin-top:2px">{condition}</div>' if condition else ''}
  <div class="ret-ev-content">{content}{'...' if len(ev.get('content','')) > 200 else ''}</div>
</div>""")
    return f"""
<div class="evidence-section" style="margin-top:12px">
  <div class="evidence-label">Retrieved evidence ({len(ev_list)})</div>
  <div class="ret-evidence-list">{''.join(items)}</div>
</div>
"""


def render_findings(findings: list, images_dir: Path) -> str:
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
    {render_finding_image_evidence(img_ev, images_dir)}
    {render_finding_ret_evidence(ret_ev)}
  </div>
</div>""")
    return f"""
<div class="section">
  <div class="section-title">Findings ({len(findings)})</div>
  <div class="findings-grid">{''.join(cards)}</div>
</div>
"""


def render_impressions(impressions: list) -> str:
    if not impressions:
        return ""
    cards = []
    for imp in impressions:
        iid = escape(imp.get("impression_id", ""))
        text = escape(imp.get("text", ""))
        linked = imp.get("linked_findings", [])
        conf = imp.get("confidence", 0.0)
        links_html = ""
        if linked:
            links_html = f'<div class="impression-links">Linked findings: {", ".join(escape(x) for x in linked)}</div>'
        cards.append(f"""
<div class="impression-card">
  <div class="impression-id">{iid} <span style="font-weight:400;color:#64748b;font-size:0.8rem">conf {conf:.2f}</span></div>
  <div class="impression-text">{text}</div>
  {links_html}
</div>""")
    return f"""
<div class="section">
  <div class="section-title">Impressions ({len(impressions)})</div>
  {''.join(cards)}
</div>
"""


def render_all_image_evidence(img_ev_list: list, images_dir: Path) -> str:
    if not img_ev_list:
        return ""
    items = []
    for ev in img_ev_list:
        img_id = ev.get("image_id", "")
        desc = ev.get("description", img_id)
        finding_desc = ev.get("finding_description", "") or desc
        tools = " → ".join(ev.get("tool_chain", []))
        p = img_path(img_id, images_dir)

        if p:
            items.append(f"""
<div class="img-thumb-wrap" onclick="openLightbox('images/{img_id}.png', `{escape(finding_desc)}`)" title="{escape(desc)}\n{escape(tools)}">
  <img class="img-thumb" src="{p}" alt="{escape(img_id)}" loading="lazy">
  <div class="img-thumb-label">{escape(img_id)}<br><span style="color:#4b5563;font-size:0.6rem">{escape(tools[:30])}</span></div>
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
    <div class="img-evidence-row" style="flex-wrap:wrap">{''.join(items)}</div>
  </div>
</div>
"""


def render_all_retrieved_evidence(ret_ev_list: list) -> str:
    if not ret_ev_list:
        return ""
    sorted_ev = sorted(ret_ev_list, key=lambda x: -x.get("relevance_score", 0))
    rows = []
    for ev in sorted_ev:
        eid = escape(ev.get("evidence_id", ""))
        src = ev.get("source_type", "")
        score = ev.get("relevance_score", 0.0)
        meta = ev.get("metadata", {})
        condition = escape(meta.get("condition", "") or meta.get("title", "") or "")
        source_ref = str(meta.get("source", ""))
        content = escape(ev.get("content", "")[:300])
        rows.append(f"""
<tr>
  <td style="white-space:nowrap">{eid}</td>
  <td>{chip(src)}</td>
  <td>{score_bar(score)}</td>
  <td>{condition}</td>
  <td style="color:#9ca3af">{content}{'...' if len(ev.get('content','')) > 300 else ''}</td>
  <td style="font-size:0.72rem;color:#4b5563">{escape(source_ref[:60])}</td>
</tr>""")

    return f"""
<div class="section">
  <div class="section-title">All Retrieved Evidence ({len(ret_ev_list)})</div>
  <div style="overflow-x:auto">
    <table class="ev-table">
      <thead>
        <tr>
          <th>ID</th><th>Source</th><th>Score</th>
          <th>Condition / Title</th><th>Content</th><th>Reference</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
"""


def render_issues(issues: list) -> str:
    if not issues:
        return ""
    cards = []
    for iss in issues:
        sev = iss.get("severity", "medium")
        iid = escape(iss.get("issue_id", ""))
        itype = escape(iss.get("issue_type", ""))
        attr = escape(iss.get("attribution", ""))
        desc = escape(iss.get("description", ""))
        fix = escape(iss.get("suggested_fix", ""))
        cards.append(f"""
<div class="issue-card sev-{sev}">
  <div class="issue-top">
    <span class="issue-id">{iid}</span>
    <span class="badge badge-{'fail' if sev in ('high','critical') else 'warn'}">{sev}</span>
    <span class="issue-type">{itype}</span>
    <span class="issue-attr">→ {attr}</span>
  </div>
  <div class="issue-desc">{desc}</div>
  {f'<div class="issue-fix">Suggested fix: {fix}</div>' if fix else ''}
</div>""")
    return f"""
<div class="section">
  <div class="section-title">Remaining Review Issues ({len(issues)})</div>
  {''.join(cards)}
</div>
"""


# ---------------------------------------------------------------------------
# Lightbox JS (simple — uses relative image paths)
# ---------------------------------------------------------------------------

LIGHTBOX_JS = """
function openLightbox(src, caption) {
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox-caption').textContent = caption;
    document.getElementById('lightbox').classList.add('open');
}
document.getElementById('lightbox').addEventListener('click', function() {
    this.classList.remove('open');
    document.getElementById('lightbox-img').src = '';
});
"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def generate_html(trace_dir: Path) -> str:
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

    # Original image: IMG_*_001 is always the first saved (original)
    orig_rel_path = None
    if images_dir.exists():
        for p in sorted(images_dir.glob("IMG_*_001.png")):
            orig_rel_path = f"images/{p.name}"
            break

    header_html = render_header(result, summary if summary else None)
    orig_report_html = render_original_and_report(orig_rel_path, nl_report)
    impressions_html = render_impressions(impressions)
    findings_html = render_findings(findings, images_dir)
    img_ev_html = render_all_image_evidence(img_ev_all, images_dir)
    ret_ev_html = render_all_retrieved_evidence(ret_ev_all)
    issues_html = render_issues(remaining_issues)

    agent_log_html = ""
    agent_log = result.get("agent_run_log", [])
    if agent_log:
        rows = "".join(
            f"<tr><td>{escape(e.get('agent',''))}</td>"
            f"<td>{e.get('iteration','')}</td>"
            f"<td>{e.get('output_length','')}</td></tr>"
            for e in agent_log
        )
        agent_log_html = f"""
<div class="section">
  <div class="section-title">Agent Run Log</div>
  <table class="ev-table">
    <thead><tr><th>Agent</th><th>Iteration</th><th>Output length (chars)</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadClaw Report — {escape(result.get('session_id', ''))}</title>
<style>{CSS}</style>
</head>
<body>

<div id="lightbox">
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-caption"></div>
</div>

<div class="container">
  {header_html}
  {orig_report_html}
  {impressions_html}
  {findings_html}
  {img_ev_html}
  {ret_ev_html}
  {issues_html}
  {agent_log_html}
</div>

<script>{LIGHTBOX_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from RadClaw trace directory")
    parser.add_argument("trace_dir", help="Path to trace directory (e.g. output/traces/sample_12)")
    parser.add_argument("--out", help="Output HTML file path (default: <trace_dir>/report.html)")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.is_dir():
        sys.exit(f"ERROR: {trace_dir} is not a directory")

    out_path = Path(args.out).resolve() if args.out else trace_dir / "report.html"

    print(f"Reading:  {trace_dir}")
    html = generate_html(trace_dir)
    out_path.write_text(html, encoding="utf-8")
    print(f"Written:  {out_path}")
    print(f"Size:     {len(html):,} bytes")
    if (trace_dir / "images").exists():
        n = len(list((trace_dir / "images").glob("*.png")))
        print(f"Images:   {n} PNG files in images/ (referenced by relative path)")
    else:
        print("Images:   no images/ directory — run pipeline with updated run_trace.py to capture crops")


if __name__ == "__main__":
    main()
