"""Shared visual style for every diagram generated for the README."""
import matplotlib.pyplot as plt

COLORS = {
    "bg": "#0f172a",
    "panel": "#1e293b",
    "panel_alt": "#273449",
    "accent": "#38bdf8",
    "accent2": "#a78bfa",
    "accent3": "#fbbf24",
    "success": "#34d399",
    "warning": "#f87171",
    "text": "#e2e8f0",
    "text_dim": "#94a3b8",
    "grid": "#334155",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "savefig.facecolor": COLORS["bg"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.edgecolor": COLORS["grid"],
    "xtick.color": COLORS["text_dim"],
    "ytick.color": COLORS["text_dim"],
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    if grid:
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
