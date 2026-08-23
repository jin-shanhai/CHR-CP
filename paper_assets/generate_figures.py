from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data" / "results_summary.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Inter, Segoe UI, Arial, sans-serif"
MONO = "Cascadia Mono, Consolas, monospace"

COLORS = {
    "ink": "#18212f",
    "muted": "#64748b",
    "line": "#cbd5e1",
    "panel": "#ffffff",
    "bg": "#f8fafc",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "violet": "#7c3aed",
    "pink": "#db2777",
    "slate": "#334155",
    "soft_blue": "#dbeafe",
    "soft_cyan": "#cffafe",
    "soft_green": "#dcfce7",
    "soft_amber": "#fef3c7",
    "soft_red": "#fee2e2",
    "soft_violet": "#ede9fe",
    "soft_pink": "#fce7f3",
}


def tag(name: str, body: str = "", **attrs: object) -> str:
    attr = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items() if v is not None)
    if body:
        return f"<{name} {attr}>{body}</{name}>"
    return f"<{name} {attr}/>"


def svg(width: int, height: int, body: str) -> str:
    defs = """
    <defs>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
        <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.12"/>
      </filter>
      <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
      </marker>
      <marker id="arrow-blue" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2563eb"/>
      </marker>
    </defs>
    """
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{COLORS['bg']}"/>
  {defs}
  <style>
    text {{ font-family: {FONT}; fill: {COLORS['ink']}; }}
    .mono {{ font-family: {MONO}; }}
    .small {{ font-size: 18px; fill: {COLORS['muted']}; }}
    .tiny {{ font-size: 15px; fill: {COLORS['muted']}; }}
    .title {{ font-size: 36px; font-weight: 760; }}
    .subtitle {{ font-size: 20px; fill: {COLORS['muted']}; }}
    .label {{ font-size: 20px; font-weight: 700; }}
    .caption {{ font-size: 16px; fill: {COLORS['muted']}; }}
  </style>
  {body}
</svg>
"""


def text(x: float, y: float, s: str, cls: str = "", size: int | None = None,
         weight: int | str | None = None, fill: str | None = None,
         anchor: str | None = None) -> str:
    attrs: dict[str, object] = {"x": round(x, 2), "y": round(y, 2)}
    if cls:
        attrs["class"] = cls
    if size:
        attrs["font-size"] = size
    if weight:
        attrs["font-weight"] = weight
    if fill:
        attrs["fill"] = fill
    if anchor:
        attrs["text-anchor"] = anchor
    return tag("text", escape(s), **attrs)


def multiline(x: float, y: float, lines: list[str], size: int = 18, fill: str | None = None,
              anchor: str | None = None, leading: int | None = None, weight: int | str | None = None) -> str:
    leading = leading or int(size * 1.35)
    parts = []
    for i, line in enumerate(lines):
        parts.append(text(x, y + i * leading, line, size=size, fill=fill, anchor=anchor, weight=weight))
    return "".join(parts)


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None,
         rx: int = 16, opacity: float | None = None, shadow: bool = False,
         sw: float = 1.5) -> str:
    attrs: dict[str, object] = {
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "rx": rx,
        "fill": fill,
    }
    if stroke:
        attrs["stroke"] = stroke
        attrs["stroke_width"] = sw
    if opacity is not None:
        attrs["opacity"] = opacity
    if shadow:
        attrs["filter"] = "url(#shadow)"
    return tag("rect", **attrs)


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = COLORS["line"],
         sw: float = 2, dash: str | None = None, arrow: bool = False, blue_arrow: bool = False) -> str:
    attrs: dict[str, object] = {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "stroke": stroke,
        "stroke_width": sw,
        "stroke_linecap": "round",
    }
    if dash:
        attrs["stroke_dasharray"] = dash
    if arrow:
        attrs["marker_end"] = "url(#arrow)"
    if blue_arrow:
        attrs["marker_end"] = "url(#arrow-blue)"
    return tag("line", **attrs)


def path(d: str, stroke: str = COLORS["line"], fill: str = "none", sw: float = 2,
         dash: str | None = None, arrow: bool = False, blue_arrow: bool = False) -> str:
    attrs: dict[str, object] = {
        "d": d,
        "stroke": stroke,
        "fill": fill,
        "stroke_width": sw,
        "stroke_linecap": "round",
        "stroke_linejoin": "round",
    }
    if dash:
        attrs["stroke_dasharray"] = dash
    if arrow:
        attrs["marker_end"] = "url(#arrow)"
    if blue_arrow:
        attrs["marker_end"] = "url(#arrow-blue)"
    return tag("path", **attrs)


def circle(cx: float, cy: float, r: float, fill: str, stroke: str | None = None, sw: float = 2) -> str:
    attrs: dict[str, object] = {"cx": round(cx, 2), "cy": round(cy, 2), "r": r, "fill": fill}
    if stroke:
        attrs["stroke"] = stroke
        attrs["stroke_width"] = sw
    return tag("circle", **attrs)


def pill(x: float, y: float, w: float, h: float, label: str, fill: str, stroke: str, txt: str | None = None) -> str:
    return rect(x, y, w, h, fill, stroke, rx=int(h / 2), sw=1.4) + text(x + w / 2, y + h / 2 + 7, label, size=18, weight=700, fill=txt or stroke, anchor="middle")


def card(x: float, y: float, w: float, h: float, title: str, body: list[str], accent: str,
         fill: str = COLORS["panel"]) -> str:
    out = rect(x, y, w, h, fill, "#e2e8f0", rx=18, shadow=True)
    out += rect(x, y, 8, h, accent, rx=18)
    out += text(x + 28, y + 38, title, "label", fill=COLORS["ink"])
    out += multiline(x + 28, y + 72, body, size=17, fill=COLORS["muted"], leading=25)
    return out


def figure_header(title: str, subtitle: str, width: int) -> str:
    return text(70, 72, title, "title") + text(72, 105, subtitle, "subtitle") + line(70, 128, width - 70, 128, COLORS["line"], 1.4)


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def money(v: float) -> str:
    if v < 0.01:
        return f"${v:.4f}"
    return f"${v:.3f}"


def fig_architecture() -> None:
    w, h = 1600, 1040
    out = figure_header(
        "CHR-CP System Overview",
        "Confidence-gated multi-agent routing with cache-preserved switching under API constraints",
        w,
    )
    out += card(70, 180, 260, 160, "Input Task", ["Problem text", "budget cap", "benchmark type"], COLORS["blue"])
    out += line(330, 260, 405, 260, arrow=True)
    out += card(405, 165, 300, 190, "L0 / L1 Routing", ["difficulty probe", "task category", "agent chain and start tier"], COLORS["cyan"])
    out += line(705, 260, 780, 260, arrow=True)
    out += card(780, 155, 360, 220, "L2 Step Routing", ["primary agent call", "VC2 uncertainty U", "STAY / BRANCH / ESCALATE"], COLORS["violet"])
    out += line(1140, 260, 1215, 260, arrow=True)
    out += card(1215, 165, 315, 190, "L3 Switching", ["stable prefix", "CADS distillation", "CTOR handoff"], COLORS["amber"])

    out += rect(770, 430, 780, 360, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += text(810, 480, "Step-level decision kernel", "label")
    out += pill(812, 515, 220, 48, "VC2: U signal", COLORS["soft_violet"], COLORS["violet"])
    out += pill(1060, 515, 250, 48, "CA2R thresholds", COLORS["soft_blue"], COLORS["blue"])
    out += pill(1338, 515, 160, 48, "Budget", "#f1f5f9", COLORS["slate"])
    out += line(1032, 539, 1060, 539, arrow=True)
    out += line(1310, 539, 1338, 539, arrow=True)
    out += rect(820, 605, 205, 96, COLORS["soft_green"], "#86efac", rx=16)
    out += text(922, 646, "STAY", size=25, weight=800, fill=COLORS["green"], anchor="middle")
    out += text(922, 675, "accept current tier", size=16, fill=COLORS["muted"], anchor="middle")
    out += rect(1065, 605, 205, 96, COLORS["soft_amber"], "#fcd34d", rx=16)
    out += text(1168, 646, "BRANCH", size=25, weight=800, fill=COLORS["amber"], anchor="middle")
    out += text(1168, 675, "verify / vote", size=16, fill=COLORS["muted"], anchor="middle")
    out += rect(1310, 605, 205, 96, COLORS["soft_red"], "#fecaca", rx=16)
    out += text(1412, 646, "ESCALATE", size=25, weight=800, fill=COLORS["red"], anchor="middle")
    out += text(1412, 675, "move up ladder", size=16, fill=COLORS["muted"], anchor="middle")

    out += rect(70, 430, 635, 360, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += text(110, 480, "API-visible evidence only", "label")
    out += multiline(
        110,
        525,
        [
            "No logprobs required for reasoning APIs",
            "Verbalized confidence: <confidence>X.X/10</confidence>",
            "Consistency: K lightweight verification samples",
            "Cache events: hit tokens, total tokens, provider boundary",
        ],
        size=20,
        fill=COLORS["muted"],
        leading=38,
    )
    out += pill(110, 700, 170, 48, "black-box", COLORS["soft_blue"], COLORS["blue"])
    out += pill(300, 700, 165, 48, "traceable", COLORS["soft_green"], COLORS["green"])
    out += pill(485, 700, 170, 48, "cost-aware", COLORS["soft_amber"], COLORS["amber"])

    out += path("M1375 355 C1375 420 1375 420 1375 500", COLORS["amber"], sw=3, blue_arrow=True)
    out += path("M1380 790 C1380 900 970 900 970 720", COLORS["blue"], sw=3, dash="9 8", blue_arrow=True)
    out += text(1000, 828, "cache events feed back into adaptive thresholds", size=18, fill=COLORS["blue"])
    out += line(1350, 355, 1350, 415, COLORS["amber"], sw=3, arrow=True)
    out += card(1080, 845, 450, 120, "Output Trace", ["final answer, final tier, action distribution", "cost, latency, cache statistics"], COLORS["green"])
    (OUT / "figure_01_system_architecture.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_positioning() -> None:
    w, h = 1400, 1000
    out = figure_header(
        "Positioning for PRICAI",
        "The current project is strongest as API-aware multi-agent LLM routing, not as directed fuzzing yet",
        w,
    )
    x0, y0, pw, ph = 180, 190, 1040, 660
    out += rect(x0, y0, pw, ph, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += rect(x0 + pw * 0.55, y0, pw * 0.45, ph * 0.48, COLORS["soft_blue"], None, rx=22, opacity=0.82)
    out += rect(x0 + pw * 0.55, y0, pw * 0.52, ph * 0.48, "none", None, rx=0)
    out += line(x0 + 70, y0 + ph - 70, x0 + pw - 70, y0 + ph - 70, COLORS["slate"], 2.5, arrow=True)
    out += line(x0 + 70, y0 + ph - 70, x0 + 70, y0 + 70, COLORS["slate"], 2.5, arrow=True)
    for i in range(1, 5):
        xx = x0 + 70 + i * (pw - 140) / 5
        yy = y0 + ph - 70 - i * (ph - 140) / 5
        out += line(xx, y0 + 70, xx, y0 + ph - 70, "#e2e8f0", 1)
        out += line(x0 + 70, yy, x0 + pw - 70, yy, "#e2e8f0", 1)
    out += text(x0 + pw / 2, y0 + ph - 18, "API and cache awareness", size=22, weight=760, anchor="middle")
    out += tag("text", "routing granularity", x=x0 - 70, y=y0 + ph / 2, transform=f"rotate(-90 {x0 - 70} {y0 + ph / 2})", **{"font-size": 22, "font-weight": 760})
    out += text(x0 + pw - 250, y0 + 92, "open gap", size=21, weight=760, fill=COLORS["blue"])
    out += text(x0 + pw - 250, y0 + 123, "fine-grained + API/cache-aware", size=17, fill=COLORS["blue"])
    points = DATA["positioning"]
    for item in points:
        px = x0 + 70 + item["x_api_cache_awareness"] * (pw - 140)
        py = y0 + ph - 70 - item["y_routing_granularity"] * (ph - 140)
        if item["method"] == "CHR-CP":
            out += circle(px, py, 28, COLORS["blue"], "#ffffff", 5)
            out += text(px + 40, py - 12, item["method"], size=30, weight=850, fill=COLORS["blue"])
            out += text(px + 40, py + 18, "ours", size=18, fill=COLORS["blue"])
        else:
            out += circle(px, py, 18, "#ffffff", COLORS["slate"], 3)
            out += text(px + 28, py + 7, item["method"], size=20, weight=680)
    out += rect(180, 885, 1040, 54, "#ffffff", "#e2e8f0", rx=18)
    out += text(210, 920, "Submission framing:", size=19, weight=760, fill=COLORS["ink"])
    out += text(395, 920, "Agents + Large Language Models + Cost-sensitive AI systems. Directed fuzzing needs extra coverage/crash evidence.", size=18, fill=COLORS["muted"])
    (OUT / "figure_02_pricai_positioning.svg").write_text(svg(w, h, out), encoding="utf-8")


def log_scale(value: float, low: float, high: float, start: float, end: float) -> float:
    lv, ll, lh = math.log10(value), math.log10(low), math.log10(high)
    return start + (lv - ll) / (lh - ll) * (end - start)


def fig_cost_accuracy() -> None:
    w, h = 1600, 980
    out = figure_header(
        "Cost-Accuracy Map",
        "Completed experiment results used in the manuscript",
        w,
    )
    x0, y0, pw, ph = 160, 185, 1140, 650
    out += rect(x0, y0, pw, ph, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += line(x0 + 80, y0 + ph - 75, x0 + pw - 80, y0 + ph - 75, COLORS["slate"], 2.5)
    out += line(x0 + 80, y0 + ph - 75, x0 + 80, y0 + 70, COLORS["slate"], 2.5)

    min_cost, max_cost = 0.001, 0.1
    y_min, y_max = 0.80, 1.00
    for c in [0.001, 0.003, 0.01, 0.03, 0.1]:
        xx = log_scale(c, min_cost, max_cost, x0 + 80, x0 + pw - 80)
        out += line(xx, y0 + 70, xx, y0 + ph - 75, "#e2e8f0", 1)
        out += text(xx, y0 + ph - 38, money(c), size=16, fill=COLORS["muted"], anchor="middle")
    for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
        yy = y0 + ph - 75 - (a - y_min) / (y_max - y_min) * (ph - 145)
        out += line(x0 + 80, yy, x0 + pw - 80, yy, "#e2e8f0", 1)
        out += text(x0 + 55, yy + 6, pct(a), size=16, fill=COLORS["muted"], anchor="end")

    def xy(cost: float, acc: float) -> tuple[float, float]:
        return (
            log_scale(cost, min_cost, max_cost, x0 + 80, x0 + pw - 80),
            y0 + ph - 75 - (acc - y_min) / (y_max - y_min) * (ph - 145),
        )

    for b in DATA["benchmark_snapshot"]:
        x1, y1 = xy(b["chrcp_cost"], b["chrcp_accuracy"])
        x2, y2 = xy(b["single_t4_cost"], b["single_t4_accuracy"])
        out += line(x1, y1, x2, y2, "#cbd5e1", 2, dash="5 7")
        out += circle(x2, y2, 15, COLORS["soft_red"], COLORS["red"], 3)
        out += circle(x1, y1, 17, COLORS["soft_blue"], COLORS["blue"], 3)
        out += text(x1 + 18, y1 - 12, b["benchmark"], size=17, weight=700, fill=COLORS["blue"])

    for cfg in DATA["phase1b_configs"]:
        xx, yy = xy(cfg["cost_per_sample"], cfg["accuracy"])
        fill = COLORS["green"] if cfg["config"] == "E2" else COLORS["amber"] if cfg["config"] == "D2" else "#ffffff"
        stroke = COLORS["green"] if cfg["config"] == "E2" else COLORS["amber"] if cfg["config"] == "D2" else COLORS["muted"]
        out += rect(xx - 18, yy - 18, 36, 36, fill, stroke, rx=9, sw=2.5)
        out += text(xx, yy + 7, cfg["config"], size=15, weight=800, fill="#ffffff" if cfg["config"] in {"E2", "D2"} else COLORS["slate"], anchor="middle")

    out += text(x0 + pw / 2, y0 + ph + 10, "Average cost per sample, log scale", size=20, weight=760, anchor="middle")
    out += tag("text", "accuracy", x=x0 - 80, y=y0 + ph / 2, transform=f"rotate(-90 {x0 - 80} {y0 + ph / 2})", **{"font-size": 20, "font-weight": 760})
    out += rect(1340, 245, 200, 190, "#ffffff", "#e2e8f0", rx=18, shadow=True)
    out += text(1370, 285, "Legend", size=22, weight=760)
    out += circle(1375, 325, 12, COLORS["soft_blue"], COLORS["blue"], 3)
    out += text(1400, 332, "CHR-CP", size=18)
    out += circle(1375, 365, 12, COLORS["soft_red"], COLORS["red"], 3)
    out += text(1400, 372, "Single-T4", size=18)
    out += rect(1362, 397, 26, 26, COLORS["green"], COLORS["green"], rx=7)
    out += text(1400, 418, "Phase 1b config", size=18)
    out += text(160, 900, "Interpretation: CHR-CP is attractive on MATH/HumanEval; GPQA and MMLU currently need stronger cost controls or reframing.", size=18, fill=COLORS["muted"])
    (OUT / "figure_03_cost_accuracy_map.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_phase1b() -> None:
    w, h = 1650, 980
    out = figure_header(
        "Phase 1b Sensitivity Summary",
        "E2 is the cost-oriented setting; D2 is the balanced setting with broader tier usage",
        w,
    )
    left = (90, 190, 690, 650)
    right = (835, 190, 720, 650)
    out += rect(*left, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += rect(*right, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += text(125, 238, "Accuracy and cost", "label")
    out += text(870, 238, "Routing behavior", "label")

    configs = DATA["phase1b_configs"]
    x0, y0, pw, ph = left
    plot_x, plot_y, plot_w, plot_h = x0 + 80, y0 + 95, pw - 150, ph - 190
    out += line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, COLORS["slate"], 2)
    out += line(plot_x, plot_y + plot_h, plot_x, plot_y, COLORS["slate"], 2)
    for acc in [0.94, 0.96, 0.98]:
        yy = plot_y + plot_h - (acc - 0.93) / 0.06 * plot_h
        out += line(plot_x, yy, plot_x + plot_w, yy, "#e2e8f0", 1)
        out += text(plot_x - 16, yy + 6, pct(acc), size=15, fill=COLORS["muted"], anchor="end")
    bar_gap = plot_w / len(configs)
    max_cost = max(c["cost_per_sample"] for c in configs)
    for i, cfg in enumerate(configs):
        cx = plot_x + i * bar_gap + bar_gap / 2
        acc_y = plot_y + plot_h - (cfg["accuracy"] - 0.93) / 0.06 * plot_h
        cost_h = cfg["cost_per_sample"] / max_cost * 150
        out += rect(cx - 27, plot_y + plot_h - cost_h, 54, cost_h, COLORS["soft_amber"], "#fbbf24", rx=9)
        out += circle(cx, acc_y, 17, COLORS["blue"] if cfg["config"] in {"E2", "D2"} else "#ffffff", COLORS["blue"], 3)
        out += text(cx, acc_y + 7, cfg["config"], size=14, weight=760, fill="#ffffff" if cfg["config"] in {"E2", "D2"} else COLORS["blue"], anchor="middle")
        out += text(cx, plot_y + plot_h + 34, cfg["config"], size=17, weight=760, anchor="middle")
        out += text(cx, plot_y + plot_h + 62, money(cfg["cost_per_sample"]), size=14, fill=COLORS["muted"], anchor="middle")
    out += text(plot_x + 120, plot_y + 25, "circle: accuracy", size=16, fill=COLORS["blue"])
    out += text(plot_x + 120, plot_y + 50, "bar: cost/sample", size=16, fill=COLORS["amber"])

    x0, y0, pw, ph = right
    bx, by, bw, bh = x0 + 70, y0 + 105, pw - 145, 48
    colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]
    labels = ["STAY", "BRANCH", "ESCALATE"]
    for i, cfg in enumerate(configs):
        y = by + i * 90
        out += text(bx - 20, y + 31, cfg["config"], size=19, weight=760, anchor="end")
        start = bx
        vals = [cfg["stay_pct"], cfg["branch_pct"], cfg["escalate_pct"]]
        for val, col, lab in zip(vals, colors, labels):
            ww = bw * val / 100
            out += rect(start, y, ww, bh, col, None, rx=8)
            if ww > 45:
                out += text(start + ww / 2, y + 31, f"{val}%", size=16, weight=760, fill="#ffffff", anchor="middle")
            start += ww
        tier_text = f"T2:{cfg['final_t2']}  T3:{cfg['final_t3']}  T4:{cfg['final_t4']}"
        out += text(bx, y + 70, tier_text, size=15, fill=COLORS["muted"])
    lx = x0 + 70
    for i, (lab, col) in enumerate(zip(labels, colors)):
        out += rect(lx + i * 175, y0 + ph - 60, 26, 18, col, None, rx=5)
        out += text(lx + i * 175 + 36, y0 + ph - 44, lab, size=16, fill=COLORS["muted"])
    out += rect(90, 875, 1465, 52, "#ffffff", "#e2e8f0", rx=18)
    out += text(125, 909, "Recommended paper use:", size=18, weight=760)
    out += text(330, 909, "show E2 as cost-optimal and D2 as balanced; avoid claiming one universal best setting across all benchmarks.", size=18, fill=COLORS["muted"])
    (OUT / "figure_04_phase1b_sensitivity.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_benchmark_dashboard() -> None:
    w, h = 1650, 1040
    out = figure_header(
        "Benchmark Results Dashboard",
        "Current project-status numbers: CHR-CP vs Single-T4 on accuracy and cost",
        w,
    )
    cards = DATA["benchmark_snapshot"]
    cw, ch = 285, 315
    start_x, start_y = 70, 185
    gap = 28
    for i, b in enumerate(cards):
        x = start_x + i * (cw + gap)
        y = start_y
        cheaper = b["single_t4_cost"] / b["chrcp_cost"] if b["chrcp_cost"] else 0
        out += rect(x, y, cw, ch, "#ffffff", "#e2e8f0", rx=22, shadow=True)
        out += text(x + 24, y + 44, b["benchmark"], size=26, weight=820)
        out += text(x + 24, y + 75, f"cost ratio: {cheaper:.1f}x", size=15, fill=COLORS["muted"])
        ax = x + 28
        ay = y + 118
        out += text(ax, ay - 18, "Accuracy", size=17, weight=760)
        out += rect(ax, ay, 210, 24, "#e2e8f0", None, rx=8)
        out += rect(ax, ay, 210 * b["chrcp_accuracy"], 24, COLORS["blue"], None, rx=8)
        out += text(ax + 220, ay + 19, pct(b["chrcp_accuracy"]), size=16, fill=COLORS["blue"])
        out += rect(ax, ay + 42, 210, 24, "#e2e8f0", None, rx=8)
        out += rect(ax, ay + 42, 210 * b["single_t4_accuracy"], 24, COLORS["red"], None, rx=8)
        out += text(ax + 220, ay + 61, pct(b["single_t4_accuracy"]), size=16, fill=COLORS["red"])
        cy = ay + 112
        max_cost = max(b["chrcp_cost"], b["single_t4_cost"])
        out += text(ax, cy - 18, "Cost / sample", size=17, weight=760)
        out += rect(ax, cy, 210 * b["chrcp_cost"] / max_cost, 28, COLORS["blue"], None, rx=8)
        out += text(ax + 220, cy + 21, money(b["chrcp_cost"]), size=16, fill=COLORS["blue"])
        out += rect(ax, cy + 46, 210 * b["single_t4_cost"] / max_cost, 28, COLORS["red"], None, rx=8)
        out += text(ax + 220, cy + 67, money(b["single_t4_cost"]), size=16, fill=COLORS["red"])
    out += rect(600, 565, 450, 70, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += circle(635, 600, 12, COLORS["blue"])
    out += text(657, 607, "CHR-CP", size=18, weight=760)
    out += circle(785, 600, 12, COLORS["red"])
    out += text(807, 607, "Single-T4", size=18, weight=760)
    out += rect(70, 700, 1485, 190, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += text(105, 748, "Paper message from the completed experiments", "label")
    notes = [
        "MATH and HumanEval support the cost-efficient routing story.",
        "AIME suggests routing can improve robustness, but cost is close to Single-T4.",
        "GPQA and MMLU currently weaken the universal cost-saving claim; present them as stress tests or add stronger ablations.",
    ]
    out += multiline(105, 790, notes, size=22, fill=COLORS["muted"], leading=42)
    (OUT / "figure_05_benchmark_dashboard.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_trace() -> None:
    w, h = 1600, 900
    out = figure_header(
        "Example Routing Trace",
        "A trace-level case study from the retained completed evaluation artifact",
        w,
    )
    steps = [
        ("Task", "MATH problem", "input", COLORS["blue"]),
        ("L1", "chain: solver -> verifier -> escalator", "route", COLORS["cyan"]),
        ("Step 1", "solver @ T1\nU=0.32", "BRANCH", COLORS["amber"]),
        ("Step 2", "verifier @ T2\nU=0.54", "ESCALATE", COLORS["red"]),
        ("L3", "CADS/CTOR handoff\nT2 -> T3", "switch", COLORS["violet"]),
        ("Step 3", "escalator @ T3\ncache-aware", "FINAL", COLORS["green"]),
    ]
    x_start, y, gap = 95, 360, 255
    last_cx = None
    for i, (title, detail, action, col) in enumerate(steps):
        cx = x_start + i * gap
        if last_cx is not None:
            out += line(last_cx + 74, y, cx - 74, y, COLORS["line"], 4, arrow=True)
        out += circle(cx, y, 74, "#ffffff", col, 6)
        out += text(cx, y - 10, title, size=22, weight=820, fill=col, anchor="middle")
        out += text(cx, y + 22, action, size=16, weight=760, fill=COLORS["muted"], anchor="middle")
        detail_lines = []
        for part in detail.split("\n"):
            detail_lines.extend(wrap(part, 22))
        out += multiline(cx, y + 118, detail_lines, size=17, fill=COLORS["muted"], anchor="middle", leading=25)
        last_cx = cx

    out += rect(165, 620, 1270, 110, "#ffffff", "#e2e8f0", rx=22, shadow=True)
    out += text(205, 665, "Why this figure matters", size=23, weight=820)
    out += text(205, 700, "It makes CHR-CP's decisions auditable: each action is tied to U, thresholds, tier, and handoff state.", size=20, fill=COLORS["muted"])
    out += path("M1125 434 C1125 530 1010 545 1000 620", COLORS["violet"], sw=3, dash="8 8", arrow=True)
    out += text(1030, 548, "cache-preserved switch", size=17, fill=COLORS["violet"])
    (OUT / "figure_06_decision_trace_template.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_tier_ladder() -> None:
    w, h = 1600, 980
    out = figure_header(
        "Four-Tier Model Ladder and Cache Boundaries",
        "Economic structure behind routing: capability rises, cost rises, and cache validity changes at provider boundaries",
        w,
    )
    tiers = DATA["tier_ladder"]
    x0, y0, card_w, card_h, gap = 90, 220, 320, 430, 48
    max_out = max(t["output_per_m"] for t in tiers)
    for i, t in enumerate(tiers):
        x = x0 + i * (card_w + gap)
        accent = [COLORS["green"], COLORS["cyan"], COLORS["blue"], COLORS["red"]][i]
        out += rect(x, y0, card_w, card_h, "#ffffff", "#e2e8f0", rx=24, shadow=True)
        out += rect(x, y0, card_w, 14, accent, None, rx=24)
        out += text(x + 28, y0 + 62, t["tier"], size=34, weight=880, fill=accent)
        out += text(x + 28, y0 + 98, t["model"], size=20, weight=760)
        out += text(x + 28, y0 + 128, f"{t['provider']} / {t['mode']}", size=16, fill=COLORS["muted"])
        out += text(x + 28, y0 + 185, "Input $/M", size=16, weight=760)
        out += text(x + 168, y0 + 185, f"${t['input_per_m']:.3g}", size=18, fill=COLORS["muted"])
        out += text(x + 28, y0 + 228, "Output $/M", size=16, weight=760)
        out += text(x + 168, y0 + 228, f"${t['output_per_m']:.3g}", size=18, fill=COLORS["muted"])
        bar_w = 240 * math.log10(t["output_per_m"] + 1) / math.log10(max_out + 1)
        out += rect(x + 28, y0 + 260, 240, 24, "#e2e8f0", None, rx=8)
        out += rect(x + 28, y0 + 260, bar_w, 24, accent, None, rx=8)
        disc = t["cache_discount"]
        label = "no listed discount" if disc is None else f"{int(disc * 100)}% cache discount"
        out += pill(x + 28, y0 + 326, 230, 44, label, "#f8fafc", accent, txt=accent)
    boundary_y = y0 + card_h + 74
    out += line(x0 + card_w + 17, boundary_y - 35, x0 + card_w + 17, boundary_y + 35, COLORS["red"], 3, dash="8 6")
    out += text(x0 + card_w + 17, boundary_y + 70, "cross-vendor", size=17, fill=COLORS["red"], anchor="middle")
    out += line(x0 + 2 * card_w + gap + 17, boundary_y - 35, x0 + 2 * card_w + gap + 17, boundary_y + 35, COLORS["green"], 3)
    out += text(x0 + 2 * card_w + gap + 17, boundary_y + 70, "same vendor", size=17, fill=COLORS["green"], anchor="middle")
    out += line(x0 + 3 * card_w + 2 * gap + 17, boundary_y - 35, x0 + 3 * card_w + 2 * gap + 17, boundary_y + 35, COLORS["red"], 3, dash="8 6")
    out += text(x0 + 3 * card_w + 2 * gap + 17, boundary_y + 70, "cross-vendor", size=17, fill=COLORS["red"], anchor="middle")
    out += rect(120, 825, 1360, 74, "#ffffff", "#e2e8f0", rx=22)
    out += text(155, 870, "Design implication:", size=20, weight=820)
    out += text(345, 870, "the router should not only ask whether a stronger model is needed, but also whether switching will destroy cache value.", size=20, fill=COLORS["muted"])
    (OUT / "figure_07_tier_ladder_cache.svg").write_text(svg(w, h, out), encoding="utf-8")


def fig_fuzzing_bridge() -> None:
    w, h = 1600, 960
    out = figure_header(
        "If Reframed as Multi-Agent Directed Fuzzing",
        "A bridge design: what must be added beyond the current CHR-CP implementation",
        w,
    )
    cols = [
        ("Current CHR-CP", ["LLM agents", "uncertainty routing", "cost/cache control", "benchmark QA/code tasks"], COLORS["blue"]),
        ("Missing fuzzing core", ["seed corpus", "mutation operators", "coverage distance", "crash/bug oracle"], COLORS["red"]),
        ("CHR-Fuzz target", ["planner agent", "mutator agent", "executor agent", "triage agent"], COLORS["green"]),
    ]
    for i, (title, bullets, col) in enumerate(cols):
        x = 90 + i * 500
        out += rect(x, 215, 420, 470, "#ffffff", "#e2e8f0", rx=24, shadow=True)
        out += rect(x, 215, 420, 14, col, None, rx=24)
        out += text(x + 32, 275, title, size=28, weight=850, fill=col)
        for j, b in enumerate(bullets):
            yy = 335 + j * 72
            out += circle(x + 42, yy - 7, 9, col)
            out += text(x + 66, yy, b, size=21, fill=COLORS["ink"])
    out += line(510, 450, 590, 450, COLORS["line"], 4, arrow=True)
    out += line(1010, 450, 1090, 450, COLORS["line"], 4, arrow=True)
    out += rect(150, 760, 1300, 94, "#ffffff", "#e2e8f0", rx=22)
    out += text(190, 810, "Recommendation:", size=22, weight=850)
    out += text(420, 810, "submit current work as API-aware multi-agent routing, unless you can run directed fuzzing experiments before June 13.", size=21, fill=COLORS["muted"])
    (OUT / "figure_08_fuzzing_reframe_bridge.svg").write_text(svg(w, h, out), encoding="utf-8")


def write_contact_sheet() -> None:
    files = [
        "figure_01_system_architecture.svg",
        "figure_02_pricai_positioning.svg",
        "figure_03_cost_accuracy_map.svg",
        "figure_04_phase1b_sensitivity.svg",
        "figure_05_benchmark_dashboard.svg",
        "figure_06_decision_trace_template.svg",
        "figure_07_tier_ladder_cache.svg",
        "figure_08_fuzzing_reframe_bridge.svg",
    ]
    body = "<h1>CHR-CP Paper Figures</h1>\n<p>Generated from paper_assets/data/results_summary.json.</p>\n"
    for f in files:
        body += f'<section><h2>{f}</h2><img src="figures/{f}" style="width:100%;max-width:1200px;border:1px solid #ddd"/></section>\n'
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CHR-CP paper figure contact sheet</title>
  <style>
    body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 40px; color: #18212f; background: #f8fafc; }}
    section {{ background: white; padding: 24px; margin: 28px 0; border-radius: 14px; box-shadow: 0 8px 26px rgba(15,23,42,.08); }}
    h1 {{ margin-bottom: 0; }}
    h2 {{ font-size: 18px; color: #334155; }}
  </style>
</head>
<body>{body}</body>
</html>
"""
    (ROOT / "figure_contact_sheet.html").write_text(html, encoding="utf-8")


def main() -> None:
    fig_architecture()
    fig_positioning()
    fig_cost_accuracy()
    fig_phase1b()
    fig_benchmark_dashboard()
    fig_trace()
    fig_tier_ladder()
    fig_fuzzing_bridge()
    write_contact_sheet()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
