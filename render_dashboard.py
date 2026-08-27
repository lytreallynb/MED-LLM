"""Render a standalone HTML dashboard from evaluation metrics JSON.

Reads the JSON written by `python -m medllm.evaluation --output ...` and
produces a self-contained HTML file (no external assets) that can be opened
directly in a browser or served by any static host.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MED-LLM Evaluation Dashboard</title>
<style>
  :root {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
    --series: #2a78d6; --missed: #898781;
    --border: rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
      --series: #3987e5; --missed: #898781;
      --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="light"] {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
    --series: #2a78d6; --missed: #898781;
    --border: rgba(11,11,11,0.10);
  }
  :root[data-theme="dark"] {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --series: #3987e5; --missed: #898781;
    --border: rgba(255,255,255,0.10);
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 24px; line-height: 1.45;
  }
  .wrap { max-width: 1060px; margin: 0 auto; }
  header h1 { font-size: 20px; font-weight: 650; }
  header p { color: var(--ink-2); font-size: 13px; margin-top: 4px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }
  .tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .tile .label { font-size: 12px; color: var(--ink-2); }
  .tile .value { font-size: 28px; font-weight: 650; margin-top: 2px; }
  .tile .sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 760px) { .cards { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; overflow-x: auto;
  }
  .card.wide { grid-column: 1 / -1; }
  .card h2 { font-size: 13.5px; font-weight: 600; }
  .card .note { font-size: 11.5px; color: var(--muted); margin-top: 2px; margin-bottom: 10px; }
  svg text { font-family: inherit; }
  .axis-label { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .cat-label { font-size: 12px; fill: var(--ink-2); }
  .val-label { font-size: 11px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
  .tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18); max-width: 320px;
  }
  .tooltip .t-title { font-weight: 600; margin-bottom: 2px; }
  .tooltip .t-row { color: var(--ink-2); font-variant-numeric: tabular-nums; }
  details.table-view { margin-top: 12px; }
  details.table-view summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
  th { color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--surface); }
  td.num { font-variant-numeric: tabular-nums; text-align: right; }
  th.num { text-align: right; }
  .scroll { max-height: 420px; overflow-y: auto; }
  footer { color: var(--muted); font-size: 11.5px; margin-top: 20px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>MED-LLM Evaluation Dashboard</h1>
    <p id="subtitle"></p>
  </header>
  <div class="tiles" id="tiles"></div>
  <div class="cards">
    <div class="card wide">
      <h2>Retrieval recall by section</h2>
      <div class="note">Share of questions whose source chunk was retrieved in the top k, grouped by FDA label section.</div>
      <div id="by-section"></div>
    </div>
    <div class="card">
      <h2>Rank of the expected chunk</h2>
      <div class="note">Where the answer chunk landed in the result list. Gray means it was not retrieved.</div>
      <div id="rank-dist"></div>
    </div>
    <div class="card">
      <h2>Top-1 similarity score distribution</h2>
      <div class="note">Cosine score of the best retrieved chunk per question.</div>
      <div id="score-hist"></div>
    </div>
    <div class="card wide">
      <h2>Pipeline health</h2>
      <div class="note">Aggregate answer-side metrics from the evaluation run.</div>
      <div id="health"></div>
      <details class="table-view">
        <summary>Per-question table</summary>
        <div class="scroll"><table id="detail-table"></table></div>
      </details>
    </div>
  </div>
  <footer id="footer"></footer>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA_JSON__;
const GENERATED_AT = "__GENERATED_AT__";

const result = DATA.results[0];
const details = (DATA.details && DATA.details[result.dataset]) || [];
const cfg = DATA.config || {};
const topK = cfg.top_k || 4;

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }
function el(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const key in attrs) {
    const value = attrs[key];
    // Route color tokens through style so theme toggles restyle live charts
    if ((key === "fill" || key === "stroke") && String(value).startsWith("var(")) node.style[key] = value;
    else node.setAttribute(key, value);
  }
  return node;
}
const tooltip = document.getElementById("tooltip");
function bindTip(target, html) {
  target.addEventListener("mouseenter", () => { tooltip.innerHTML = html; tooltip.style.display = "block"; });
  target.addEventListener("mousemove", (e) => {
    const pad = 14;
    let x = e.clientX + pad, y = e.clientY + pad;
    const r = tooltip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    tooltip.style.left = x + "px"; tooltip.style.top = y + "px";
  });
  target.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
}

// Header
document.getElementById("subtitle").textContent =
  `Dataset: ${result.dataset} (${result.total} questions) | Embeddings: ${cfg.embedding_model || "n/a"} | ` +
  `LLM: ${cfg.qwen_model || "retrieval only"} | top_k = ${topK}`;
document.getElementById("footer").textContent =
  `Generated ${GENERATED_AT} from ${result.total} evaluation questions. ` +
  `Answer accuracy requires a DASHSCOPE_API_KEY; retrieval metrics are computed locally.`;

// Stat tiles
const tiles = [
  { label: `Recall@${topK}`, value: fmtPct(result.recall_at_k), sub: "expected chunk retrieved" },
  { label: "MRR", value: result.mrr.toFixed(3), sub: "mean reciprocal rank" },
  { label: "Retrieval hit rate", value: fmtPct(result.retrieval_hit_rate), sub: "answer text in evidence" },
  { label: "Avg latency", value: Math.round(result.avg_latency_ms) + " ms", sub: "per query, end to end" },
];
document.getElementById("tiles").innerHTML = tiles.map(t =>
  `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="sub">${t.sub}</div></div>`
).join("");

// Horizontal bar helper with rounded data-end
function hBarPath(x, y, w, h, r) {
  if (w <= r) r = Math.max(0, w - 0.5);
  return `M${x},${y} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${h - 2 * r} a${r},${r} 0 0 1 -${r},${r} h-${w - r} z`;
}
function vBarPath(x, yTop, w, h, r) {
  if (h <= r) r = Math.max(0, h - 0.5);
  return `M${x},${yTop + r} a${r},${r} 0 0 1 ${r},-${r} h${w - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${h - r} h-${w} z`;
}

// Chart 1: recall by section (horizontal bars, one measure, one hue)
(function bySection() {
  const groups = {};
  for (const row of details) {
    const key = row.section || "unknown";
    if (!groups[key]) groups[key] = { n: 0, recalled: 0, scoreSum: 0 };
    groups[key].n += 1;
    if (row.rank > 0) groups[key].recalled += 1;
    groups[key].scoreSum += row.top_score;
  }
  const items = Object.entries(groups).map(([section, g]) => ({
    section, n: g.n, rate: g.recalled / g.n, avgScore: g.scoreSum / g.n,
  })).sort((a, b) => b.rate - a.rate);
  if (!items.length) return;

  const labelW = 210, chartW = 640, rowH = 30, barH = 16, padTop = 6;
  const width = labelW + chartW + 60, height = padTop + items.length * rowH + 24;
  const svg = el("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": "Retrieval recall by FDA label section" });
  // gridlines at 0/25/50/75/100%
  for (let i = 0; i <= 4; i++) {
    const gx = labelW + (chartW * i) / 4;
    svg.appendChild(el("line", { x1: gx, y1: padTop, x2: gx, y2: height - 24,
      stroke: "var(--grid)", "stroke-width": 1 }));
    const tick = el("text", { x: gx, y: height - 8, "text-anchor": "middle", class: "axis-label" });
    tick.textContent = (i * 25) + "%";
    svg.appendChild(tick);
  }
  items.forEach((item, i) => {
    const y = padTop + i * rowH + (rowH - barH) / 2;
    const w = Math.max(1, chartW * item.rate);
    const label = el("text", { x: labelW - 10, y: y + barH - 4, "text-anchor": "end", class: "cat-label" });
    label.textContent = item.section.replaceAll("_", " ");
    svg.appendChild(label);
    const bar = el("path", { d: hBarPath(labelW, y, w, barH, 4), fill: "var(--series)" });
    svg.appendChild(bar);
    const val = el("text", { x: labelW + w + 8, y: y + barH - 4, class: "val-label" });
    val.textContent = fmtPct(item.rate);
    svg.appendChild(val);
    const hit = el("rect", { x: 0, y: padTop + i * rowH, width: width, height: rowH, fill: "transparent" });
    bindTip(hit, `<div class="t-title">${item.section.replaceAll("_", " ")}</div>` +
      `<div class="t-row">recall@${topK}: ${fmtPct(item.rate)}</div>` +
      `<div class="t-row">questions: ${item.n}</div>` +
      `<div class="t-row">avg top score: ${item.avgScore.toFixed(3)}</div>`);
    svg.appendChild(hit);
  });
  document.getElementById("by-section").appendChild(svg);
})();

// Chart 2: rank distribution (vertical bars; missed shown as gray state)
(function rankDist() {
  const counts = [];
  for (let r = 1; r <= topK; r++) counts.push({ label: "rank " + r, n: 0, missed: false });
  counts.push({ label: "missed", n: 0, missed: true });
  for (const row of details) {
    if (row.rank > 0 && row.rank <= topK) counts[row.rank - 1].n += 1;
    else counts[counts.length - 1].n += 1;
  }
  const maxN = Math.max(1, ...counts.map(c => c.n));
  const width = 440, height = 220, padL = 36, padB = 26, padT = 14;
  const plotW = width - padL - 12, plotH = height - padT - padB;
  const slot = plotW / counts.length, barW = Math.min(44, slot - 10);
  const svg = el("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": "Rank distribution of the expected chunk" });
  for (let i = 0; i <= 3; i++) {
    const gy = padT + (plotH * i) / 3;
    svg.appendChild(el("line", { x1: padL, y1: gy, x2: width - 12, y2: gy,
      stroke: "var(--grid)", "stroke-width": 1 }));
    const tick = el("text", { x: padL - 6, y: gy + 4, "text-anchor": "end", class: "axis-label" });
    tick.textContent = Math.round(maxN * (1 - i / 3));
    svg.appendChild(tick);
  }
  svg.appendChild(el("line", { x1: padL, y1: padT + plotH, x2: width - 12, y2: padT + plotH,
    stroke: "var(--baseline)", "stroke-width": 1 }));
  counts.forEach((c, i) => {
    const x = padL + i * slot + (slot - barW) / 2;
    const h = Math.max(c.n > 0 ? 2 : 0, plotH * (c.n / maxN));
    const yTop = padT + plotH - h;
    if (h > 0) {
      svg.appendChild(el("path", { d: vBarPath(x, yTop, barW, h, 4),
        fill: c.missed ? "var(--missed)" : "var(--series)" }));
    }
    const val = el("text", { x: x + barW / 2, y: yTop - 5, "text-anchor": "middle", class: "val-label" });
    if (c.n > 0) val.textContent = c.n;
    svg.appendChild(val);
    const lab = el("text", { x: x + barW / 2, y: height - 8, "text-anchor": "middle", class: "axis-label" });
    lab.textContent = c.label;
    svg.appendChild(lab);
    const hit = el("rect", { x: padL + i * slot, y: padT, width: slot, height: plotH + padB, fill: "transparent" });
    bindTip(hit, `<div class="t-title">${c.label}</div><div class="t-row">${c.n} question(s)</div>`);
    svg.appendChild(hit);
  });
  document.getElementById("rank-dist").appendChild(svg);
})();

// Chart 3: top score histogram
(function scoreHist() {
  const scores = details.map(d => d.top_score).filter(s => s > 0);
  if (!scores.length) return;
  const lo = Math.floor(Math.min(...scores) * 20) / 20;
  const hi = Math.ceil(Math.max(...scores) * 20) / 20;
  const binW = 0.05;
  const nBins = Math.max(1, Math.round((hi - lo) / binW));
  const bins = Array.from({ length: nBins }, (_, i) => ({
    from: lo + i * binW, to: lo + (i + 1) * binW, n: 0,
  }));
  for (const s of scores) {
    let idx = Math.min(nBins - 1, Math.floor((s - lo) / binW));
    bins[idx].n += 1;
  }
  const maxN = Math.max(1, ...bins.map(b => b.n));
  const width = 440, height = 220, padL = 36, padB = 26, padT = 14;
  const plotW = width - padL - 12, plotH = height - padT - padB;
  const slot = plotW / nBins, barW = Math.max(4, slot - 2);
  const svg = el("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": "Distribution of top-1 similarity scores" });
  for (let i = 0; i <= 3; i++) {
    const gy = padT + (plotH * i) / 3;
    svg.appendChild(el("line", { x1: padL, y1: gy, x2: width - 12, y2: gy,
      stroke: "var(--grid)", "stroke-width": 1 }));
    const tick = el("text", { x: padL - 6, y: gy + 4, "text-anchor": "end", class: "axis-label" });
    tick.textContent = Math.round(maxN * (1 - i / 3));
    svg.appendChild(tick);
  }
  svg.appendChild(el("line", { x1: padL, y1: padT + plotH, x2: width - 12, y2: padT + plotH,
    stroke: "var(--baseline)", "stroke-width": 1 }));
  bins.forEach((b, i) => {
    const x = padL + i * slot + (slot - barW) / 2;
    const h = plotH * (b.n / maxN);
    const yTop = padT + plotH - h;
    if (b.n > 0) svg.appendChild(el("path", { d: vBarPath(x, yTop, barW, h, 3), fill: "var(--series)" }));
    const hit = el("rect", { x: padL + i * slot, y: padT, width: slot, height: plotH + padB, fill: "transparent" });
    bindTip(hit, `<div class="t-title">score ${b.from.toFixed(2)} to ${b.to.toFixed(2)}</div>` +
      `<div class="t-row">${b.n} question(s)</div>`);
    svg.appendChild(hit);
    if (i % Math.ceil(nBins / 6) === 0) {
      const lab = el("text", { x: padL + i * slot, y: height - 8, "text-anchor": "middle", class: "axis-label" });
      lab.textContent = b.from.toFixed(2);
      svg.appendChild(lab);
    }
  });
  document.getElementById("score-hist").appendChild(svg);
})();

// Pipeline health strip
(function health() {
  const rows = [
    { label: "Grounding correctness", value: fmtPct(result.grounding_correctness), note: "queries answered from evidence (not refused for low similarity)" },
    { label: "Completeness", value: fmtPct(result.completeness), note: "queries with at least 2 supporting chunks" },
    { label: "Abstention rate", value: fmtPct(result.hallucination_rate), note: "queries refused for weak evidence (reported as hallucination_rate)" },
    { label: "Answer accuracy", value: cfg.qwen_model ? fmtPct(result.accuracy) : "n/a", note: cfg.qwen_model ? "answer contains the expected key" : "needs DASHSCOPE_API_KEY (run without --no-qwen)" },
    { label: "Avg top score", value: result.avg_top_score.toFixed(3), note: "mean cosine similarity of best hit" },
  ];
  document.getElementById("health").innerHTML =
    '<div class="tiles" style="margin:8px 0 0">' + rows.map(r =>
      `<div class="tile"><div class="label">${r.label}</div><div class="value" style="font-size:22px">${r.value}</div><div class="sub">${r.note}</div></div>`
    ).join("") + "</div>";
})();

// Per-question table
(function table() {
  const table = document.getElementById("detail-table");
  const header = "<tr><th>Question</th><th>Section</th><th class='num'>Rank</th><th class='num'>Top score</th><th class='num'>Latency (ms)</th></tr>";
  const body = details.map(d =>
    `<tr><td>${d.question}</td><td>${(d.section || "").replaceAll("_", " ")}</td>` +
    `<td class="num">${d.rank > 0 ? d.rank : "miss"}</td>` +
    `<td class="num">${d.top_score.toFixed(3)}</td>` +
    `<td class="num">${Math.round(d.latency_ms)}</td></tr>`
  ).join("");
  table.innerHTML = header + body;
})();
</script>
</body>
</html>
"""


def render(metrics_path: Path, output_path: Path) -> Path:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not payload.get("results"):
        raise ValueError(f"No results found in {metrics_path}")
    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/metrics.json", help="Metrics JSON from medllm.evaluation")
    parser.add_argument("--output", default="results/dashboard.html", help="Where to write the dashboard HTML")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    path = render(Path(args.metrics), Path(args.output))
    print(f"Dashboard written to {path}")
