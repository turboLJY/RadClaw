#!/usr/bin/env python3
"""
RadClaw Demo Server — browse reports in a web UI.

Works behind any reverse proxy (DSW, JupyterHub, ngrok, etc.) because
all navigation URLs are built client-side from window.location.

Usage:
    python scripts/demo_server.py --traces output/traces_demo --port 8899
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from scripts.generate_report_html import (
    CSS, render_header, render_impressions, render_issues,
    render_all_retrieved_evidence, chip, score_bar,
)

app = FastAPI(title="RadClaw Demo")
TRACES_DIR: Path = Path("output/traces")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_summary(sample_dir: Path) -> dict:
    p = sample_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    p = sample_dir / "result.json"
    if p.exists():
        r = json.loads(p.read_text())
        return {
            "iterations": r.get("iterations"),
            "review_passed": r.get("review_passed"),
            "image_evidence_count": r.get("image_evidence_count", 0),
            "retrieved_evidence_count": r.get("retrieved_evidence_count", 0),
        }
    return {}


def _generate_nl_report(result: dict) -> str:
    """Generate a clean FINDINGS/IMPRESSION report from structured_report."""
    sr = result.get("structured_report") or {}
    findings = sr.get("findings", [])
    impressions = sr.get("impressions", [])
    if not findings and not impressions:
        return result.get("natural_language_report", "(no report)")

    lines = []
    if findings:
        lines.append("FINDINGS:")
        for f in findings:
            fid = f.get("finding_id", "")
            text = f.get("text", "")
            lines.append(f"  [{fid}] {text}")
        lines.append("")
    if impressions:
        lines.append("IMPRESSION:")
        for i, imp in enumerate(impressions, 1):
            lines.append(f"  {i}. {imp.get('text', '')}")
    return "\n".join(lines)


def _load_image_index(sample: str) -> dict:
    """Load image_index.json for crop_box data."""
    p = TRACES_DIR / sample / "image_index.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

@app.get("/images/{sample}/{img_file}")
def get_image(sample: str, img_file: str):
    p = TRACES_DIR / sample / "images" / img_file
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(404)
    return Response(content=p.read_bytes(), media_type="image/png")


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

INDEX_JS = """
(function() {
  var base = window.location.href.replace(/\\/?$/, '/');
  document.querySelectorAll('[data-sample]').forEach(function(el) {
    el.href = base + 'report/' + el.dataset.sample;
  });
})();
"""

INDEX_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
.container { max-width: 960px; margin: 0 auto; padding: 40px 24px; }
h1 { color: #7ec8e3; font-size: 2rem; margin-bottom: 8px; }
.subtitle { color: #64748b; margin-bottom: 32px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
.card {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px;
    padding: 20px; text-decoration: none; color: inherit;
    transition: border-color .2s, transform .15s; display: block; cursor: pointer;
}
.card:hover { border-color: #7ec8e3; transform: translateY(-2px); }
.card-title { font-size: 1.1rem; font-weight: 700; color: #7ec8e3; margin-bottom: 10px; }
.card-row { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 700; }
.pass { background: #1a3a2a; color: #4ade80; }
.fail { background: #3a1a1a; color: #f87171; }
"""


@app.get("/", response_class=HTMLResponse)
def index():
    samples = sorted(
        [d for d in TRACES_DIR.iterdir() if d.is_dir() and (d / "result.json").exists()],
        key=lambda d: d.name,
    )
    cards = ""
    for d in samples:
        s = _load_summary(d)
        passed = s.get("review_passed")
        badge = (
            '<span class="badge pass">✓ Passed</span>' if passed is True
            else '<span class="badge fail">✗ Not Passed</span>' if passed is False
            else ""
        )
        elapsed = s.get("elapsed_seconds", "")
        iters = s.get("iterations", "?")
        img_cnt = s.get("image_evidence_count", "?")
        ret_cnt = s.get("retrieved_evidence_count", "?")
        cards += f"""
<a class="card" data-sample="{d.name}" href="#">
  <div class="card-title">{d.name}</div>
  <div class="card-row">{badge}</div>
  <div class="card-row">{iters} iteration(s){f" · {elapsed}s" if elapsed else ""}</div>
  <div class="card-row">&#128247; {img_cnt} image &nbsp;·&nbsp; &#128196; {ret_cnt} retrieved</div>
</a>"""

    if not cards:
        cards = f'<p style="color:#4b5563;font-style:italic">No samples found in {TRACES_DIR}</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>RadClaw Demo</title>
<style>{INDEX_CSS}</style></head>
<body>
<div class="container">
  <h1>RadClaw Demo</h1>
  <p class="subtitle">Multi-agent radiology report generation &mdash; select a sample</p>
  <div class="grid">{cards}</div>
</div>
<script>{INDEX_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Report page JS — builds URLs from window.location, renders markdown
# ---------------------------------------------------------------------------

REPORT_JS = r"""
// Base URL detection (works behind any reverse proxy)
var base = window.location.href.replace(/\/report\/[^/?#]+.*$/, '');

// Back link
var bl = document.getElementById('back-link');
if (bl) bl.href = base + '/';

// Fill image src from data-img-id attributes
document.querySelectorAll('img[data-img-id]').forEach(function(img) {
  var src = base + '/images/' + img.dataset.sample + '/' + img.dataset.imgId + '.png';
  img.src = src;
  img.style.cursor = 'pointer';
  img.onclick = function() { openLightbox(src, img.dataset.caption || ''); };
});

// Lightbox
function openLightbox(src, caption) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-caption').textContent = caption;
  document.getElementById('lightbox').classList.add('open');
}
document.getElementById('lightbox').addEventListener('click', function() {
  this.classList.remove('open');
  document.getElementById('lightbox-img').src = '';
});

// Markdown rendering (simple: bold, code, bullet)
function renderMd(text) {
  return text
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code style="background:#1e2535;padding:1px 5px;border-radius:3px;font-size:0.85em">$1</code>')
    .replace(/^#{1,3} (.+)$/gm, '<div style="font-weight:700;color:#93c5fd;margin:8px 0 4px">$1</div>')
    .replace(/^[-*] (.+)$/gm, '<div style="padding-left:16px;margin:2px 0">· $1</div>');
}
document.querySelectorAll('.md-render').forEach(function(el) {
  el.innerHTML = renderMd(el.textContent);
});

// Show more / less for evidence content
document.querySelectorAll('.ev-toggle').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var target = document.getElementById(this.dataset.target);
    if (target.style.maxHeight === 'none') {
      target.style.maxHeight = '80px';
      this.textContent = 'show more ▾';
    } else {
      target.style.maxHeight = 'none';
      this.textContent = 'show less ▴';
    }
  });
});

// Canvas annotation on original image
var canvas = document.getElementById('annot-canvas');
if (canvas) {
  var origImg = document.getElementById('orig-img');
  var hoveredBox = -1;

  function setupCanvas() {
    if (!origImg.naturalWidth) return;
    canvas.width = origImg.naturalWidth;
    canvas.height = origImg.naturalHeight;
    requestAnimationFrame(function() { drawAnnotations(hoveredBox); });
  }
  origImg.addEventListener('load', setupCanvas);
  if (origImg.complete && origImg.naturalWidth > 0) setupCanvas();

  // Hover: highlight box under cursor
  canvas.addEventListener('mousemove', function(e) {
    var idx = hitBox(e);
    if (idx !== hoveredBox) {
      hoveredBox = idx;
      canvas.style.cursor = idx >= 0 ? 'pointer' : 'crosshair';
      drawAnnotations(hoveredBox);
    }
  });
  canvas.addEventListener('mouseleave', function() {
    hoveredBox = -1;
    drawAnnotations(-1);
  });

  // Click: open lightbox with the evidence image
  canvas.addEventListener('click', function(e) {
    var idx = hitBox(e);
    if (idx < 0) return;
    var b = (window.CROP_BOXES || [])[idx];
    if (!b || !b.img_id) return;
    var src = base + '/images/' + b.sample + '/' + b.img_id + '.png';
    openLightbox(src, b.fid + ': ' + (b.label || ''));
  });

  function hitBox(e) {
    if (!canvas.width) return -1;
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var cx = (e.clientX - rect.left) * scaleX;
    var cy = (e.clientY - rect.top) * scaleY;
    var boxes = window.CROP_BOXES || [];
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i];
      if (cx >= b.x && cx <= b.x + b.w && cy >= b.y && cy <= b.y + b.h) return i;
    }
    return -1;
  }
}

function drawAnnotations(highlightIdx) {
  var canvas = document.getElementById('annot-canvas');
  if (!canvas || !canvas.width) return;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  var boxes = window.CROP_BOXES || [];
  var colors = {'abnormal': '#f87171', 'normal': '#4ade80', 'uncertain': '#fbbf24'};
  // Font size: aim for ~18px visible; scale to canvas coordinate space
  var fontSize = Math.round(canvas.width / 40);
  ctx.font = 'bold ' + fontSize + 'px sans-serif';

  boxes.forEach(function(b, i) {
    var color = colors[b.type] || '#7ec8e3';
    var isHot = (i === highlightIdx);
    // Fill
    ctx.globalAlpha = isHot ? 0.30 : 0.12;
    ctx.fillStyle = color;
    ctx.fillRect(b.x, b.y, b.w, b.h);
    // Border
    ctx.globalAlpha = isHot ? 1.0 : 0.85;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(4, Math.round(canvas.width / 500)) * (isHot ? 2 : 1);
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    // Label background + text
    ctx.globalAlpha = 1.0;
    var label = b.fid + (b.label ? ': ' + b.label.substring(0, 35) : '');
    var tw = ctx.measureText(label).width;
    var lx = b.x;
    var ly = b.y > fontSize + 4 ? b.y - 4 : b.y + b.h + fontSize + 4;
    ctx.fillStyle = 'rgba(0,0,0,0.72)';
    ctx.fillRect(lx, ly - fontSize, tw + fontSize * 0.6, fontSize * 1.4);
    ctx.fillStyle = color;
    ctx.fillText(label, lx + fontSize * 0.3, ly);
    // Hint icon when hovered
    if (isHot) {
      var hint = '🔍 click to view';
      ctx.font = Math.round(fontSize * 0.75) + 'px sans-serif';
      var hw = ctx.measureText(hint).width;
      ctx.fillStyle = 'rgba(0,0,0,0.72)';
      ctx.fillRect(b.x + b.w - hw - fontSize, ly - fontSize, hw + fontSize * 0.6, fontSize * 1.4);
      ctx.fillStyle = '#e0e0e0';
      ctx.fillText(hint, b.x + b.w - hw - fontSize * 0.7, ly);
      ctx.font = 'bold ' + fontSize + 'px sans-serif';
    }
  });
}
"""

NAV_CSS = """
.nav {
    background: #1a1f2e; border-bottom: 1px solid #2d3748;
    padding: 12px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100;
}
.nav a { color: #7ec8e3; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
.nav a:hover { color: #fff; }
.nav-sep { color: #374151; }
.nav-title { color: #94a3b8; font-size: 0.9rem; }
"""

ANNOT_CSS = """
.orig-wrap { position: relative; display: inline-block; width: 100%; }
.orig-wrap img { width: 100%; display: block; border-radius: 6px; }
.orig-wrap canvas {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    cursor: crosshair; border-radius: 6px;
}
"""

EXTRA_CSS = """
.ev-content { max-height: 80px; overflow: hidden; transition: max-height 0.2s; }
.ev-toggle {
    background: none; border: none; color: #4b5563; cursor: pointer;
    font-size: 0.72rem; padding: 2px 0; margin-top: 2px;
}
.ev-toggle:hover { color: #94a3b8; }
.report-box { white-space: pre-wrap; }
.md-render { white-space: normal; line-height: 1.7; }
"""


def _img_thumb(img_id: str, sample: str, caption: str, tools: str = "") -> str:
    return (
        f'<div class="img-thumb-wrap">'
        f'<img class="img-thumb" data-img-id="{img_id}" data-sample="{sample}"'
        f' data-caption="{caption}" src="" alt="{img_id}" loading="lazy">'
        f'<div class="img-thumb-label">{img_id}'
        + (f'<br><span style="color:#4b5563;font-size:0.6rem">{tools[:28]}</span>' if tools else "")
        + "</div></div>"
    )


def _ev_item(ev: dict, ev_idx: int) -> str:
    eid = ev.get("evidence_id", "")
    src_type = ev.get("source_type", "")
    score = ev.get("relevance_score", 0.0)
    content = ev.get("content", "")
    meta = ev.get("metadata", {})
    condition = meta.get("condition", "") or meta.get("title", "")
    uid = f"ev-{eid}-{ev_idx}".replace(" ", "")
    needs_toggle = len(content) > 200

    content_html = (
        f'<div class="ev-content" id="{uid}" style="max-height:{"none" if not needs_toggle else "80px"}">'
        f'<div class="md-render">{content}</div></div>'
    )
    toggle_html = (
        f'<button class="ev-toggle" data-target="{uid}">show more ▾</button>'
        if needs_toggle else ""
    )
    return f"""
<div class="ret-ev-item">
  <span class="ret-ev-id">{eid}</span>{chip(src_type)}
  <span class="ret-ev-score">{score:.3f}</span>
  {f'<div style="color:#7ec8e3;font-size:0.75rem;margin-top:2px">{condition}</div>' if condition else ""}
  {content_html}{toggle_html}
</div>"""


def render_findings_server(findings: list, sample: str) -> str:
    if not findings:
        return ""
    cards = []
    for f in findings:
        fid = f.get("finding_id", "")
        ftype = f.get("type", "abnormal")
        anatomy = f.get("anatomy", "")
        obs = f.get("observation", "")
        text = f.get("text", "")
        conf = f.get("confidence", 0.0)
        support = f.get("support", {})
        img_ev = support.get("image_evidence", [])
        ret_ev = support.get("retrieved_evidence", [])
        type_cls = f"type-{ftype}" if ftype in ("normal", "abnormal", "uncertain") else "type-abnormal"
        obs_html = f'<span style="color:#94a3b8;font-size:0.8rem"> · {obs}</span>' if obs else ""

        img_items = "".join(
            _img_thumb(ev.get("image_id",""), sample,
                       ev.get("finding_description","") or ev.get("description",""),
                       " → ".join(ev.get("tool_chain",[])))
            for ev in img_ev
        )
        img_ev_html = f"""
<div class="evidence-section">
  <div class="evidence-label">Image evidence ({len(img_ev)})</div>
  <div class="img-evidence-row">{img_items}</div>
</div>""" if img_items else ""

        ret_items = "".join(_ev_item(ev, i) for i, ev in enumerate(ret_ev))
        ret_ev_html = f"""
<div class="evidence-section" style="margin-top:12px">
  <div class="evidence-label">Retrieved evidence ({len(ret_ev)})</div>
  <div class="ret-evidence-list">{ret_items}</div>
</div>""" if ret_items else ""

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
    {img_ev_html}{ret_ev_html}
  </div>
</div>""")

    return f"""
<div class="section">
  <div class="section-title">Findings ({len(findings)})</div>
  <div class="findings-grid">{"".join(cards)}</div>
</div>"""


def render_all_img_ev_server(img_ev_list: list, sample: str) -> str:
    if not img_ev_list:
        return ""
    items = "".join(
        _img_thumb(ev.get("image_id",""), sample,
                   ev.get("finding_description","") or ev.get("description",""),
                   " → ".join(ev.get("tool_chain",[])))
        for ev in img_ev_list
    )
    return f"""
<div class="section">
  <div class="section-title">All Image Evidence ({len(img_ev_list)})</div>
  <div class="image-panel">
    <div class="img-evidence-row" style="flex-wrap:wrap">{items}</div>
  </div>
</div>"""


def render_all_ret_ev_server(ret_ev_list: list) -> str:
    if not ret_ev_list:
        return ""
    sorted_ev = sorted(ret_ev_list, key=lambda x: -x.get("relevance_score", 0))
    rows = []
    for ev in sorted_ev:
        eid = ev.get("evidence_id", "")
        src = ev.get("source_type", "")
        score = ev.get("relevance_score", 0.0)
        meta = ev.get("metadata", {})
        condition = meta.get("condition", "") or meta.get("title", "")
        content = ev.get("content", "")
        source_ref = str(meta.get("source", ""))[:60]

        uid = f"allev-{eid}".replace(" ", "")
        needs_toggle = len(content) > 300
        content_html = (
            f'<div id="{uid}" style="color:#9ca3af;font-size:0.75rem;'
            f'max-height:{"none" if not needs_toggle else "60px"};overflow:hidden">{content}</div>'
        )
        toggle_html = (
            f'<button class="ev-toggle" data-target="{uid}">show more ▾</button>'
            if needs_toggle else ""
        )
        rows.append(f"""
<tr>
  <td style="white-space:nowrap">{eid}</td>
  <td>{chip(src)}</td>
  <td>{score_bar(score)}</td>
  <td>{condition}</td>
  <td>{content_html}{toggle_html}</td>
  <td style="font-size:0.72rem;color:#4b5563">{source_ref}</td>
</tr>""")

    return f"""
<div class="section">
  <div class="section-title">All Retrieved Evidence ({len(ret_ev_list)})</div>
  <div style="overflow-x:auto">
    <table class="ev-table">
      <thead><tr><th>ID</th><th>Source</th><th>Score</th>
      <th>Condition / Title</th><th>Content</th><th>Reference</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</div>"""


@app.get("/report/{sample}", response_class=HTMLResponse)
def report(sample: str):
    result_path = TRACES_DIR / sample / "result.json"
    if not result_path.exists():
        raise HTTPException(404, detail=f"Sample '{sample}' not found")

    result = json.loads(result_path.read_text())
    summary_path = TRACES_DIR / sample / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    image_index = _load_image_index(sample)

    sr = result.get("structured_report") or {}
    findings = sr.get("findings", [])
    impressions = sr.get("impressions", [])
    nl_report = _generate_nl_report(result)
    img_ev_all = result.get("image_evidence", [])
    ret_ev_all = result.get("retrieved_evidence", [])
    remaining_issues = result.get("remaining_issues", [])

    # Build CROP_BOXES JS data for canvas annotation
    # Map image_id -> crop_box from image_index
    img_meta_map = {img_id: data.get("metadata", {}) for img_id, data in image_index.items()}

    # Build per-finding crop box list
    crop_boxes_js = []
    finding_type_map = {f.get("finding_id", ""): f.get("type", "abnormal") for f in findings}
    for f in findings:
        fid = f.get("finding_id", "")
        ftype = f.get("type", "abnormal")
        for ev in f.get("support", {}).get("image_evidence", []):
            img_id = ev.get("image_id", "")
            meta = img_meta_map.get(img_id, {})
            cb = meta.get("crop_box") or ev.get("crop_box")
            if cb:
                safe_label = ev.get("finding_description", "")[:50].replace('"', "").replace("'", "")
                crop_boxes_js.append(
                    f'{{fid:"{fid}",type:"{ftype}",label:"{safe_label}",'
                    f'img_id:"{img_id}",sample:"{sample}",'
                    f'x:{cb["x"]},y:{cb["y"]},w:{cb["width"]},h:{cb["height"]}}}'
                )

    crop_boxes_str = "[\n" + ",\n".join(crop_boxes_js) + "\n]"

    # Original image (IMG_*_001)
    orig_id = ""
    images_dir = TRACES_DIR / sample / "images"
    if images_dir.exists():
        for p in sorted(images_dir.glob("IMG_*_001.png")):
            orig_id = p.stem
            break

    orig_html = ""
    if orig_id:
        orig_html = f"""
<div class="section">
  <div class="section-title">Original Radiograph
    <span style="font-size:0.72rem;font-weight:400;color:#4b5563;margin-left:8px">
      colored boxes = image evidence regions
    </span>
  </div>
  <div class="image-panel">
    <div class="orig-wrap">
      <img id="orig-img" data-img-id="{orig_id}" data-sample="{sample}"
           data-caption="Original Radiograph" src="" alt="Original X-ray">
      <canvas id="annot-canvas"></canvas>
    </div>
    <div class="image-caption">Click image to enlarge &nbsp;·&nbsp; colored boxes show cropped regions</div>
  </div>
</div>"""

    report_html = f"""
<div class="section">
  <div class="section-title">Radiology Report</div>
  <div class="report-box">{nl_report}</div>
</div>"""

    top_html = (
        f'<div class="main-layout">{orig_html}{report_html}</div>'
        if orig_html else report_html
    )

    agent_log_rows = "".join(
        f"<tr><td>{e.get('agent','')}</td><td>{e.get('iteration','')}</td>"
        f"<td>{e.get('output_length','')}</td></tr>"
        for e in result.get("agent_run_log", [])
    )
    agent_log_html = f"""
<div class="section">
  <div class="section-title">Agent Run Log</div>
  <table class="ev-table">
    <thead><tr><th>Agent</th><th>Iteration</th><th>Output length (chars)</th></tr></thead>
    <tbody>{agent_log_rows}</tbody>
  </table>
</div>""" if agent_log_rows else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadClaw &mdash; {sample}</title>
<style>{NAV_CSS}{CSS}{ANNOT_CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="nav">
  <a id="back-link" href="#">&larr; All Samples</a>
  <span class="nav-sep">|</span>
  <span class="nav-title">{sample}</span>
</div>
<div id="lightbox">
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-caption"></div>
</div>
<div class="container">
  {render_header(result, summary or None)}
  {top_html}
  {render_impressions(impressions)}
  {render_findings_server(findings, sample)}
  {render_all_img_ev_server(img_ev_all, sample)}
  {render_all_ret_ev_server(ret_ev_all)}
  {render_issues(remaining_issues)}
  {agent_log_html}
</div>
<script>
window.CROP_BOXES = {crop_boxes_str};
{REPORT_JS}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RadClaw demo web server")
    parser.add_argument("--traces", default="output/traces", help="Traces directory")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    args = parser.parse_args()

    global TRACES_DIR
    TRACES_DIR = Path(args.traces).resolve()
    if not TRACES_DIR.exists():
        sys.exit(f"ERROR: {TRACES_DIR} not found")

    print(f"\nRadClaw Demo Server")
    print(f"  Traces : {TRACES_DIR}")
    print(f"  Port   : {args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
