"""Plot the fitted f4 reference-similarity weights of the configured run.

Reads ``weights.tsv`` and ``run.json`` from the run directory and generates a
stacked-bar SVG: one bar per target sample, grouped by
population, one color per fitted source.

Run with ``python plot_weights.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


# --- Configuration for the completed run ---
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUN_DIRECTORY = SCRIPT_DIRECTORY / "runs" / "example"
WEIGHTS_FILE = RUN_DIRECTORY / "weights.tsv"
RUN_FILE = RUN_DIRECTORY / "run.json"
OUTPUT_FILE = RUN_DIRECTORY / "reference_similarity.svg"

TITLE = "F4 reference similarities"
FIGSIZE = (20.0, 6.0)  # inches at 72 px/inch
LABEL_FONTSIZE = 10.0
TITLE_FONTSIZE = 12.0
LEGEND_FONTSIZE = 10.0
TEXT_WIDTH_FACTOR = 0.62  # conservative average glyph width in ems
MIN_GROUP_SIZE = 1
GROUP_GAP = 2.0
BAR_WIDTH = 1.0
OVERWRITE = True

# Display-only names. The underlying population labels remain unchanged.
DISPLAY_LABEL_OVERRIDES = {
    "Czechia_EBA_CordedWare": "Corded Ware EBA",
}

# Matplotlib tab10/tab20 hex values, matching the PopStruct plot style.
TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
TAB20 = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
)
FONT_FAMILY = "DejaVu Sans, Helvetica, Arial, sans-serif"


def load_weights() -> tuple[pd.DataFrame, list[str]]:
    """Return sample-by-source weights (file order) and the source order."""

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Run weights file not found: {WEIGHTS_FILE}")
    if RUN_FILE.exists():
        sources = [str(s) for s in json.loads(RUN_FILE.read_text())["sources"]]
    else:
        sources = []

    weights = pd.read_csv(WEIGHTS_FILE, sep="\t")
    if "target" not in weights.columns:
        raise ValueError(f"{WEIGHTS_FILE} must contain a target column")
    if not sources:
        sources = [str(column) for column in weights.columns if column != "target"]
    missing = sorted(set(sources) - set(weights.columns))
    if missing:
        raise ValueError(f"{WEIGHTS_FILE} is missing source columns: {missing}")
    unknown = sorted(set(weights.columns) - {"target", *sources})
    if unknown:
        raise ValueError(f"{WEIGHTS_FILE} contains unexpected columns: {unknown}")
    wide = weights[["target", *sources]].copy().set_index("target")

    wide.index = wide.index.astype(str)
    values = wide.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Weights contain missing or non-finite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Weights must be between zero and one")

    wide = wide.reset_index()
    # Target labels are "population:iid"; group bars by the population part.
    wide["population"] = [label.split(":", 1)[0] for label in wide["target"]]
    order = {p: i for i, p in enumerate(dict.fromkeys(wide["population"]))}
    wide = wide.sort_values("population", key=lambda s: s.map(order), kind="stable")
    return wide.reset_index(drop=True), sources


def group_layout(labels: pd.Series) -> tuple[np.ndarray, list[tuple[str, int, float]]]:
    """Bar positions with a gap between population blocks, plus block midpoints."""

    positions = np.empty(len(labels), dtype=float)
    groups: list[tuple[str, int, float]] = []
    values = labels.astype(str).to_numpy()
    cursor = 0.0
    start = 0
    while start < len(values):
        label = values[start]
        stop = start + 1
        while stop < len(values) and values[stop] == label:
            stop += 1
        count = stop - start
        positions[start:stop] = cursor + np.arange(count, dtype=float)
        groups.append((label, count, (positions[start] + positions[stop - 1]) / 2.0))
        cursor = positions[stop - 1] + 1.0 + GROUP_GAP
        start = stop
    return positions, groups


def estimated_text_width(text: str, fontsize: float) -> float:
    """Estimate text width without requiring a font-rendering library."""

    return len(text) * fontsize * TEXT_WIDTH_FACTOR


def display_label(label: str) -> str:
    """Return a readable display label without changing data identifiers."""

    label = str(label)
    return DISPLAY_LABEL_OVERRIDES.get(label, label.replace("_", " "))


def render_svg(wide: pd.DataFrame, sources: list[str]) -> str:
    base_width = FIGSIZE[0] * 72.0
    base_height = FIGSIZE[1] * 72.0
    left, right = 0.06 * base_width, 0.90 * base_width
    top = 0.08 * base_height
    plot_height = 0.68 * base_height
    bottom = top + plot_height
    y_max = 1.02

    colors = TAB10 if len(sources) <= 10 else TAB20
    if len(sources) > len(colors):
        raise ValueError(f"Too many sources to color distinctly: {len(sources)}")

    positions, groups = group_layout(wide["population"])
    if len(positions) == 0:
        raise ValueError("Cannot plot an empty weights table")

    # The x labels are rotated -45 degrees. Their diagonal extent determines
    # the required bottom margin. This keeps long population labels visible.
    longest_group_label = max(
        (display_label(label) for label, _, _ in groups),
        key=len,
        default="",
    )
    rotated_label_width = estimated_text_width(longest_group_label, LABEL_FONTSIZE)
    rotated_label_depth = rotated_label_width / np.sqrt(2.0)
    label_y = bottom + 14.0
    height = max(
        base_height,
        label_y + rotated_label_depth + LABEL_FONTSIZE + 12.0,
    )

    # Grow the canvas so the widest legend entry is never clipped.
    legend_label_width = max(
        estimated_text_width(display_label(source), LEGEND_FONTSIZE)
        for source in ["Source", *sources]
    )
    legend_x = right + 8.0
    canvas_width = max(base_width, legend_x + 14.0 + legend_label_width + 12.0)

    x_lo, x_hi = positions[0] - 0.5, positions[-1] + 0.5

    def x(value: float) -> float:
        return left + (value - x_lo) / (x_hi - x_lo) * (right - left)

    def y(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:g}" height="{height:g}" '
        f'viewBox="0 0 {canvas_width:g} {height:g}" font-family="{FONT_FAMILY}">',
        f'<rect width="{canvas_width:g}" height="{height:g}" fill="white"/>',
        f'<text x="{(left + right) / 2:.2f}" y="{top - 10:.2f}" font-size="{TITLE_FONTSIZE:g}" '
        f'text-anchor="middle" fill="black">{escape(TITLE)}</text>',
    ]

    # Stacked sample bars. Adjacent bars share rounded edge coordinates and are
    # drawn without antialiasing, so no sub-pixel seams appear between them.
    parts.append('<g shape-rendering="crispEdges">')
    values = wide[sources].to_numpy(float)
    for i, position in enumerate(positions):
        bar_left = round(x(position - BAR_WIDTH / 2.0), 2)
        bar_right = round(x(position + BAR_WIDTH / 2.0), 2)
        stacked = 0.0
        for k in range(len(sources)):
            value = values[i, k]
            if value <= 0.0:
                continue
            y_lo = round(y(stacked), 2)
            y_hi = round(y(stacked + value), 2)
            parts.append(
                f'<rect x="{bar_left:.2f}" y="{y_hi:.2f}" width="{bar_right - bar_left:.2f}" '
                f'height="{y_lo - y_hi:.2f}" fill="{colors[k]}"/>'
            )
            stacked += value
    parts.append("</g>")

    # Left spine, y ticks and label
    parts.append(
        f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{left:.2f}" y2="{y(y_max):.2f}" '
        f'stroke="black" stroke-width="0.8"/>'
    )
    for tick in np.linspace(0.0, 1.0, 6):
        parts.append(
            f'<text x="{left - 6:.2f}" y="{y(tick):.2f}" font-size="10" '
            f'text-anchor="end" dominant-baseline="middle" fill="black">{tick:.1f}</text>'
        )
    parts.append(
        f'<text x="{left - 44:.2f}" y="{(bottom + y(y_max)) / 2:.2f}" font-size="12" '
        f'text-anchor="middle" fill="black" '
        f'transform="rotate(-90 {left - 44:.2f} {(bottom + y(y_max)) / 2:.2f})">'
        f"Ancestry proportion</text>"
    )

    # Population block ticks and slanted labels
    for label, count, midpoint in groups:
        if count < MIN_GROUP_SIZE:
            continue
        tick_x = x(midpoint)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{bottom:.2f}" x2="{tick_x:.2f}" '
            f'y2="{bottom + 5:.2f}" stroke="black" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{tick_x:.2f}" y="{label_y:.2f}" font-size="{LABEL_FONTSIZE:g}" '
            f'text-anchor="end" fill="black" '
            f'transform="rotate(-45 {tick_x:.2f} {label_y:.2f})">'
            f'{escape(display_label(label))}</text>'
        )

    # Legend
    legend_y = y(y_max) + 4
    parts.append(
        f'<text x="{legend_x:.2f}" y="{legend_y:.2f}" font-size="{LEGEND_FONTSIZE:g}" '
        f'dominant-baseline="hanging" fill="black">Source</text>'
    )
    for k, source in enumerate(sources):
        entry_y = legend_y + 16 + k * 15
        parts.append(
            f'<rect x="{legend_x:.2f}" y="{entry_y:.2f}" width="10" height="10" '
            f'fill="{colors[k]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14:.2f}" y="{entry_y + 5:.2f}" font-size="{LEGEND_FONTSIZE:g}" '
            f'dominant-baseline="middle" fill="black">'
            f'{escape(display_label(source))}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> Path:
    wide, sources = load_weights()
    svg = render_svg(wide, sources)
    output = OUTPUT_FILE if OVERWRITE else unique_output_path(OUTPUT_FILE)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"Plotted {len(wide)} samples, {len(sources)} sources.")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    main()
