import matplotlib as mpl

# ─── Thesis palette ──────────────────────────────────────────────────────────
# The Python twin of utils/thesis_palette.R — same ColorBrewer Set2 for categorical
# encoding, Blues sequential, PuOr diverging. Keep the two files in step: a hue changed
# here must be changed there too, or the R and Python figures drift apart.
#
# Encoding rules
#   region   -> hue        one Set2 hue per region
#   session  -> lightness  Post = the true Set2 hue, Pre = that hue tinted toward white
#   arm      -> hue        only where no region is colour-coded, so no clash
SET2 = {
    "teal": "#66C2A5", "orange": "#FC8D62", "blue": "#8DA0CB", "pink": "#E78AC3",
    "lime": "#A6D854", "yellow": "#FFD92F", "tan": "#E5C494", "grey": "#B3B3B3",
}
PRE_TINT = 0.55


def _mix_to(hex_colour, target, f):
    """Mix hex_colour toward target by fraction f (0 = unchanged, 1 = target)."""
    a = [int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(target.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))


def tint(hex_colour, f=PRE_TINT):
    """Lighter: mix toward white."""
    return _mix_to(hex_colour, "#FFFFFF", f)


def shade(hex_colour, f=0.30):
    """Darker: mix toward black. Used for box edges and marker outlines."""
    return _mix_to(hex_colour, "#000000", f)


# Region -> hue. Cognition sits on the lime rather than Set2's yellow, which is too pale
# on white and whose Pre tint all but vanishes in print.
REGION_HUE = {
    "intero":   SET2["pink"],
    "extero":   SET2["orange"],
    "mental":   SET2["lime"],
    "visual":   SET2["blue"],
    "auditory": SET2["teal"],
}
REGION_SESSION = {k: {"Pre": tint(v), "Post": v} for k, v in REGION_HUE.items()}

# Arm. Orange = DMT-harmine, blue = placebo.
ARM_HUE = {"placebo": SET2["blue"], "verum": SET2["orange"]}
ARM_SESSION = {
    "placebo_pre":  tint(ARM_HUE["placebo"]), "placebo_post": ARM_HUE["placebo"],
    "verum_pre":    tint(ARM_HUE["verum"]),   "verum_post":   ARM_HUE["verum"],
}

# Single-series marks (histograms, one-group distributions): no categorical meaning, so
# one fixed hue rather than borrowing a region's.
MONO = {"fill": SET2["blue"], "edge": shade(SET2["blue"])}

NEUTRAL = {"grid": "#E8E8E8", "muted": SET2["grey"], "line": "#737373", "ink": "black"}

# Continuous ramps. PuOr orientation: orange = positive, purple = negative.
SEQ_CMAP = "Blues"
DIV_CMAP = "PuOr"

# Retired Palette A — kept so any unconverted script still imports cleanly.
PALETTE = {
    "group_A_pre":  "#3D5A6C",   # slate, darker — ses-01 / pre-retreat
    "group_A_post": "#7A95A6",   # slate, lighter — ses-02 / post-retreat
    "group_B_pre":  "#8B7355",   # taupe, darker
    "group_B_post": "#BFA888",   # taupe, lighter
}


def apply_thesis_style():
    """Apply thesis-wide rcParams for publication figures."""
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.6,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "axes.grid":            False,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "savefig.facecolor":    "white",
        "legend.frameon":       False,
        "axes.titleweight":     "normal",
        "axes.labelweight":     "normal",
    })
