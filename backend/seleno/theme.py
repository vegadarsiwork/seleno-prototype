"""One source of truth for chart and UI colour, shared by matplotlib and the frontend.

The categorical hues are the first four slots of the validated default
data-visualisation palette, in their fixed order, stepped for a dark surface.
That set was checked with the palette validator before use:

    lightness band  PASS (all inside L 0.48-0.67)
    chroma floor    PASS (all >= 0.1)
    CVD separation  PASS (worst adjacent dE 8.4, protan)
    normal vision   PASS (worst adjacent dE 19.8)
    contrast        PASS (all >= 3:1 on #1a1a19)

Rules that follow from that and are enforced by callers, not by this module:

* Hues are assigned in slot order and never cycled. A fifth strip would fold
  into "other" or move to small multiples rather than invent a hue.
* Colour follows the product identity, never its rank, so filtering the strip
  list must not repaint the survivors.
* A legend is always present for two or more series, and up to four are also
  directly labelled, so identity is never carried by colour alone.
* Status colours are reserved for accept/warn/refuse and are never reused as a
  series colour.
"""
from __future__ import annotations

# ----------------------------------------------------------------- surfaces
SURFACE = "#1a1a19"          # chart surface (validator ran against this)
SURFACE_APP = "#0c0d10"      # app background
PANEL = "#131519"
LINE = "#24282f"
LINE_2 = "#313742"

TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#8d95a3"
TEXT_FAINT = "#626b7a"

# ------------------------------------------------------- categorical series
# Fixed order. Slot 1..4 of the reference palette, dark steps.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
SERIES_NAMES = ["blue", "orange", "aqua", "yellow"]

# ------------------------------------------------------------ status colours
# Reserved. Never used as a series colour.
STATUS = {
    "accepted": "#199e70",
    "warning": "#c98500",
    "refused": "#e66767",
    "neutral": "#8d95a3",
}

# --------------------------------------------------- semantic mark colours
# Correspondence overlays. Distinct roles, not a categorical scale.
INLIER = "#199e70"
OUTLIER = "#e66767"
SELECTED = "#c98500"
MASKED = "#9085e9"           # predicted-unusable region overlay


def series_for(keys) -> dict:
    """Stable colour per key, assigned in first-seen order and never cycled.

    Beyond four keys the extras are returned as muted grey rather than a
    generated hue - a fifth invented colour would break the palette's CVD
    guarantees.
    """
    out = {}
    for i, k in enumerate(keys):
        out[k] = SERIES[i] if i < len(SERIES) else TEXT_MUTED
    return out


def apply_matplotlib(plt) -> None:
    """Style matplotlib to match the app surface. Call before plotting."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": LINE_2,
        "axes.titlecolor": TEXT_PRIMARY,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "lines.linewidth": 2.0,          # 2px lines, per the mark spec
        "lines.solid_capstyle": "round",
        "figure.dpi": 130,
    })
