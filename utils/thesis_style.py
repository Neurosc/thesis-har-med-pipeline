import matplotlib as mpl

# Palette A (approved)
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
