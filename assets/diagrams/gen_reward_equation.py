"""Generates assets/diagrams/reward_equation.png -- the reward architecture, as a formatted equation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _style import COLORS

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)
ax.axis("off")

ax.text(6, 6.05, "Reward Architecture", ha="center", fontsize=17, fontweight="bold", color=COLORS["text"])
ax.text(6, 5.55, "extrinsic profit signal blended with an intrinsic fairness signal",
        ha="center", fontsize=10.5, color=COLORS["text_dim"])

y_eq = 4.55
fs = 26
x = 1.2
def eq_text(s, color, weight="normal", size=fs):
    global x
    t = ax.text(x, y_eq, s, fontsize=size, color=color, fontweight=weight, va="center")
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    width_data = inv.transform((bbox.width, 0))[0] - inv.transform((0, 0))[0]
    x += width_data + 0.05

eq_text("r", COLORS["text"], "bold")
eq_text("  =  ", COLORS["text_dim"])
eq_text("r", COLORS["accent"], "bold")
eq_text("ext", COLORS["accent"], "bold", size=15)
eq_text("   +   ", COLORS["text_dim"])
eq_text("ω", COLORS["accent3"], "bold")
eq_text(" · ", COLORS["text_dim"])
eq_text("r", COLORS["accent2"], "bold")
eq_text("int", COLORS["accent2"], "bold", size=15)

def panel(x0, y0, w, h, title, formula, desc, edge):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         linewidth=1.8, edgecolor=edge, facecolor=COLORS["panel"])
    ax.add_patch(p)
    ax.text(x0 + w / 2, y0 + h - 0.35, title, ha="center", fontsize=12.5, fontweight="bold", color=edge)
    ax.text(x0 + w / 2, y0 + h - 0.85, formula, ha="center", fontsize=10.5, color=COLORS["text"], family="monospace")
    ax.text(x0 + w / 2, y0 + 0.35, desc, ha="center", fontsize=9, color=COLORS["text_dim"], wrap=True)

panel(0.6, 1.9, 5.3, 2.1, "Extrinsic — profit-driven, per delivery",
      "r_ext = payout − time_penalty·t − fuel_cost·d",
      "Rewards completing valuable, efficient deliveries.\nApplied once, when a delivery completes.",
      COLORS["accent"])

panel(6.1, 1.9, 5.3, 2.1, "Intrinsic — fairness-driven, per step",
      "r_int = coverage_bonus · (1 − zone_variance)",
      "Rewards keeping order completions evenly\nspread across the city's zones. Applied every step.",
      COLORS["accent2"])

ax.text(6, 1.55, "Behavioral penalties (subtracted from every step's reward)",
        ha="center", fontsize=10.5, fontweight="bold", color=COLORS["warning"])

penalties = [
    ("Cherry-picking", "rejecting a below-average\noffer while idle nearby"),
    ("Zone abandonment", "leaving a zone with zero\ncompletions while others thrive"),
    ("Unsafe shortcut", "choosing the risky, faster\nmovement action variant"),
]
pw = 3.75
for i, (name, desc) in enumerate(penalties):
    x0 = 0.6 + i * (pw + 0.15)
    p = FancyBboxPatch((x0, 0.15), pw, 1.15, boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.4, edgecolor=COLORS["warning"], facecolor=COLORS["panel_alt"])
    ax.add_patch(p)
    ax.text(x0 + pw / 2, 1.0, f"− {name}", ha="center", fontsize=10.5, fontweight="bold", color=COLORS["warning"])
    ax.text(x0 + pw / 2, 0.5, desc, ha="center", fontsize=8.3, color=COLORS["text_dim"])

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_equation.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved {out_path}")
