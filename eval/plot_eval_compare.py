"""
Plot evaluation comparison (DDPM 10 steps vs FM 10 steps vs FM 1 step) as SVG.

This avoids matplotlib/PIL dependency issues by generating a standalone SVG.

Usage:
    python eval/plot_eval_compare.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class EvalResult:
    label: str
    mse_mean: float
    mse_std: float
    div_mean: float
    div_std: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _nice_max(value: float) -> float:
    """
    Produce a "nice" axis max >= value.
    Keeps it simple: 1/2/5 * 10^k.
    """
    if value <= 0:
        return 1.0
    import math

    k = math.floor(math.log10(value))
    base = 10 ** k
    scaled = value / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def _ticks(max_val: float, n: int = 5) -> List[float]:
    if n <= 1:
        return [0.0, max_val]
    step = max_val / (n - 1)
    return [i * step for i in range(n)]


def _bar_chart_svg(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    results: List[Tuple[str, float, float]],
    value_note: str,
    colors: List[str],
) -> str:
    """
    results: [(label, mean, std), ...]
    Draws 0..axis_max. Includes error bars.
    """
    assert len(results) == len(colors)
    axis_padding_left = 54
    axis_padding_right = 16
    axis_padding_top = 28
    axis_padding_bottom = 46

    plot_x0 = x + axis_padding_left
    plot_y0 = y + axis_padding_top
    plot_w = w - axis_padding_left - axis_padding_right
    plot_h = h - axis_padding_top - axis_padding_bottom

    max_val = 0.0
    for _, mean, std in results:
        max_val = max(max_val, mean + std)
    axis_max = _nice_max(max_val * 1.08)
    axis_max = max(axis_max, 1e-6)

    def y_of(val: float) -> float:
        val = _clamp(val, 0.0, axis_max)
        return plot_y0 + plot_h * (1.0 - (val / axis_max))

    n = len(results)
    gap = 18
    bar_w = (plot_w - gap * (n - 1)) / n if n > 0 else plot_w

    def x_of(i: int) -> float:
        return plot_x0 + i * (bar_w + gap)

    parts: List[str] = []

    # Title
    parts.append(
        f'<text x="{x + 10:.1f}" y="{y + 18:.1f}" font-size="14" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
        f'fill="#111827">{_svg_escape(title)}</text>'
    )
    parts.append(
        f'<text x="{x + w - 10:.1f}" y="{y + 18:.1f}" text-anchor="end" '
        f'font-size="11" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
        f'fill="#6B7280">{_svg_escape(value_note)}</text>'
    )

    # Axes + grid
    ticks = _ticks(axis_max, n=5)
    for t in ticks:
        yy = y_of(t)
        parts.append(f'<line x1="{plot_x0:.1f}" y1="{yy:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{yy:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        parts.append(
            f'<text x="{plot_x0 - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="10" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
            f'fill="#374151">{t:.3g}</text>'
        )

    # Axis lines
    parts.append(f'<line x1="{plot_x0:.1f}" y1="{plot_y0:.1f}" x2="{plot_x0:.1f}" y2="{plot_y0 + plot_h:.1f}" stroke="#111827" stroke-width="1.2"/>')
    parts.append(f'<line x1="{plot_x0:.1f}" y1="{plot_y0 + plot_h:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{plot_y0 + plot_h:.1f}" stroke="#111827" stroke-width="1.2"/>')

    # Bars
    for i, (label, mean, std) in enumerate(results):
        bx = x_of(i)
        by = y_of(mean)
        bbase = y_of(0.0)
        bh = max(0.0, bbase - by)
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'rx="6" ry="6" fill="{colors[i]}" opacity="0.88"/>'
        )

        # Error bar
        top = y_of(mean + std)
        bot = y_of(max(0.0, mean - std))
        cx = bx + bar_w / 2.0
        parts.append(f'<line x1="{cx:.1f}" y1="{top:.1f}" x2="{cx:.1f}" y2="{bot:.1f}" stroke="#111827" stroke-width="1.2"/>')
        cap = min(10.0, bar_w * 0.35)
        parts.append(f'<line x1="{(cx - cap):.1f}" y1="{top:.1f}" x2="{(cx + cap):.1f}" y2="{top:.1f}" stroke="#111827" stroke-width="1.2"/>')
        parts.append(f'<line x1="{(cx - cap):.1f}" y1="{bot:.1f}" x2="{(cx + cap):.1f}" y2="{bot:.1f}" stroke="#111827" stroke-width="1.2"/>')

        # Value label
        parts.append(
            f'<text x="{cx:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
            f'font-size="10" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
            f'fill="#111827">{mean:.3f}±{std:.3f}</text>'
        )

        # X label
        parts.append(
            f'<text x="{cx:.1f}" y="{plot_y0 + plot_h + 18:.1f}" text-anchor="middle" '
            f'font-size="11" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
            f'fill="#111827">{_svg_escape(label)}</text>'
        )

    return "\n".join(parts)


def build_svg(results: List[EvalResult]) -> str:
    width = 980
    height = 620
    pad = 24
    card_gap = 18
    card_w = width - pad * 2
    card_h = (height - pad * 2 - card_gap) / 2

    bg = "#F9FAFB"
    card = "#FFFFFF"
    stroke = "#E5E7EB"

    colors = ["#2563EB", "#16A34A", "#F97316"]  # blue, green, orange
    labels = [r.label for r in results]

    mse_triplets = [(r.label, r.mse_mean, r.mse_std) for r in results]
    div_triplets = [(r.label, r.div_mean, r.div_std) for r in results]

    card1_x = pad
    card1_y = pad
    card2_x = pad
    card2_y = pad + card_h + card_gap

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')

    # Header
    parts.append(
        '<text x="24" y="34" font-size="18" '
        'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
        'fill="#111827">Evaluation Comparison</text>'
    )
    parts.append(
        '<text x="24" y="54" font-size="12" '
        'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
        'fill="#6B7280">DDPM (10 steps) vs Flow Matching (10 steps / 1 step)</text>'
    )

    # Cards
    parts.append(f'<rect x="{card1_x}" y="{card1_y + 60}" width="{card_w}" height="{card_h - 60}" rx="14" ry="14" fill="{card}" stroke="{stroke}"/>')
    parts.append(f'<rect x="{card2_x}" y="{card2_y}" width="{card_w}" height="{card_h}" rx="14" ry="14" fill="{card}" stroke="{stroke}"/>')

    parts.append(
        _bar_chart_svg(
            x=card1_x,
            y=card1_y + 60,
            w=card_w,
            h=card_h - 60,
            title="Conditional MSE (lower is better)",
            results=mse_triplets,
            value_note="mean ± std",
            colors=colors,
        )
    )
    parts.append(
        _bar_chart_svg(
            x=card2_x,
            y=card2_y,
            w=card_w,
            h=card_h,
            title="Diversity (L2) (higher is better)",
            results=div_triplets,
            value_note="mean ± std",
            colors=colors,
        )
    )

    # Legend
    legend_x = width - 24
    legend_y = 50
    for i, label in enumerate(labels):
        ly = legend_y + i * 16
        parts.append(f'<rect x="{legend_x - 220}" y="{ly - 10}" width="10" height="10" rx="2" ry="2" fill="{colors[i]}" opacity="0.88"/>')
        parts.append(
            f'<text x="{legend_x - 205}" y="{ly - 1}" font-size="11" '
            f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" '
            f'fill="#111827">{_svg_escape(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    # Values taken from terminal output:
    # - DDPM 10 steps: Conditional MSE 0.435937 ± 1.117376; Diversity 1.194873 ± 0.529813
    # - FM 10 steps:   Conditional MSE 0.460957 ± 0.998276; Diversity 10.407434 ± 8.664867
    # - FM 1 step:     Conditional MSE 0.395806 ± 1.049517; Diversity 1.081288 ± 0.933927
    results = [
        EvalResult("DDPM 10 steps", mse_mean=0.435937, mse_std=1.117376, div_mean=1.194873, div_std=0.529813),
        EvalResult("FM 10 steps", mse_mean=0.460957, mse_std=0.998276, div_mean=10.407434, div_std=8.664867),
        EvalResult("FM 1 step", mse_mean=0.395806, mse_std=1.049517, div_mean=1.081288, div_std=0.933927),
    ]

    svg = build_svg(results)
    out_path = Path(__file__).resolve().parent.parent / "images" / "eval_compare.svg"
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
