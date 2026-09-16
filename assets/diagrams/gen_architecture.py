"""Generates assets/diagrams/architecture.png -- the system architecture overview."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import COLORS

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, label, sublabel="", color=COLORS["panel"], edge=COLORS["accent"], text_color=COLORS["text"]):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=text_color)
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                 fontsize=8.5, color=COLORS["text_dim"])
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=text_color)


def arrow(x1, y1, x2, y2, color=COLORS["text_dim"], style="-|>", lw=1.6, curve=0.0):
    connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                          linewidth=lw, color=color, connectionstyle=connectionstyle)
    ax.add_patch(a)


ax.text(6.5, 7.6, "marl-fleet-fairness-eval — System Architecture",
         ha="center", fontsize=16, fontweight="bold", color=COLORS["text"])

box(0.4, 6.2, 3.2, 0.9, "NYC TLC Trip Data", "7.6M real taxi trips (2019-01)", color=COLORS["panel_alt"], edge=COLORS["accent3"])
arrow(2.0, 6.2, 2.0, 5.55, color=COLORS["accent3"])

box(0.4, 4.75, 3.2, 0.7, "data/tlc_calibration.py", "hourly demand · fare/mile", edge=COLORS["accent3"])

box(0.4, 3.3, 4.7, 1.15, "FleetDispatchEnv", "PettingZoo ParallelEnv · partial observability\nzones · orders · drivers · rewards", edge=COLORS["accent"])
arrow(2.0, 4.75, 2.0, 4.45, color=COLORS["accent3"])

box(0.4, 1.7, 4.7, 1.15, "RLlib PPO Training", "shared policy · 8 agents\ncheckpoint save/load", edge=COLORS["accent2"])
arrow(2.75, 3.3, 2.75, 2.85, color=COLORS["accent"])

box(5.5, 1.7, 4.0, 1.15, "Evaluation Suite", "efficiency · fairness (Gini)\ndecision-quality · robustness", edge=COLORS["success"])
arrow(5.1, 3.6, 5.6, 2.6, color=COLORS["accent"], curve=-0.15)
arrow(2.75, 1.7, 2.75, 1.15, color=COLORS["accent2"])

box(9.9, 1.7, 2.8, 1.15, "Centralized\nBaseline", "Hungarian-algorithm\noptimal assignment", edge=COLORS["accent3"])
arrow(9.9, 2.27, 9.55, 2.27, color=COLORS["accent3"])

box(9.9, 3.3, 2.8, 1.15, "Fairness Metrics", "Gini coefficient\nzone coverage variance", edge=COLORS["success"])
arrow(9.5, 2.4, 9.5, 3.3, color=COLORS["success"], curve=0.0)
arrow(7.6, 2.85, 9.85, 3.7, color=COLORS["success"], curve=0.18)

box(0.4, 0.15, 4.7, 1.05, "Decision Samples", "collect_decision_samples()\nreal offer/reject snapshots", edge=COLORS["accent2"])
arrow(2.75, 1.7, 2.75, 1.2, color=COLORS["success"])

box(5.5, 0.15, 4.0, 1.05, "Streamlit Annotation\nApp", "3+ human raters\nfairness & soundness", edge=COLORS["warning"])
arrow(4.55, 0.68, 5.6, 0.68, color=COLORS["accent2"])

box(9.9, 0.15, 2.8, 1.05, "Agreement Stats", "Krippendorff's α\nCohen's κ", edge=COLORS["warning"])
arrow(9.55, 0.68, 9.95, 0.68, color=COLORS["warning"])

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved {out_path}")
