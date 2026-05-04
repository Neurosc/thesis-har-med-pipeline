import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "palette_preview.png"

PALETTES = {
    "A": ["#3D5A6C", "#7A95A6", "#8B7355", "#BFA888"],
    "B": ["#4A6670", "#8FA8B2", "#A67B5B", "#D4B896"],
    "C": ["#2C5F7C", "#6B97B0", "#9C6B47", "#C9A57B"],
}

ROLE_LABELS = ["Group A\npre", "Group A\npost", "Group B\npre", "Group B\npost"]

RECT_W   = 0.65
RECT_GAP = 0.35
STEP     = RECT_W + RECT_GAP   # 1.0 per swatch
RECT_Y0  = 0.28
RECT_H   = 0.55


def hex_to_gray(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    val = int(0.299 * r + 0.587 * g + 0.114 * b)
    return f"#{val:02X}{val:02X}{val:02X}"


def draw_cell(ax, colors, hex_labels, title, title_color="black"):
    for i, (color, hex_lbl) in enumerate(zip(colors, hex_labels)):
        x = i * STEP
        ax.add_patch(
            Rectangle((x, RECT_Y0), RECT_W, RECT_H,
                       facecolor=color, edgecolor="none")
        )
        # hex code below rectangle
        ax.text(x + RECT_W / 2, RECT_Y0 - 0.04, hex_lbl,
                ha="center", va="top", fontsize=6.5,
                fontfamily="monospace", color="#444444")
        # role label above rectangle
        ax.text(x + RECT_W / 2, RECT_Y0 + RECT_H + 0.03, ROLE_LABELS[i],
                ha="center", va="bottom", fontsize=6.5, color="#555555",
                linespacing=1.3)

    n = len(colors)
    ax.set_xlim(-0.1, n * STEP - RECT_GAP + 0.1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, color=title_color, loc="left", pad=5)
    ax.axis("off")


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(10, 7))
fig.patch.set_facecolor("white")

for row, (letter, colors) in enumerate(PALETTES.items()):
    gray_colors = [hex_to_gray(c) for c in colors]
    gray_labels = [hex_to_gray(c) for c in colors]

    draw_cell(axes[row, 0], colors,      colors,      f"Palette {letter}")
    draw_cell(axes[row, 1], gray_colors, gray_labels, "grayscale", title_color="#888888")

plt.tight_layout(h_pad=2.5, w_pad=1.5)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
