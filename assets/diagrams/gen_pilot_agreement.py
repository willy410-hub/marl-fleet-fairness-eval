"""Generates assets/diagrams/pilot_agreement.png -- the pilot study's Krippendorff's Alpha results."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from _style import COLORS, style_axis

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pilot_study", "pilot_results.json",
)
with open(RESULTS_PATH) as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(9, 5.5))

labels = ["Fairness rating", "Soundness rating"]
alphas = [results["fairness_agreement"]["krippendorff_alpha"], results["soundness_agreement"]["krippendorff_alpha"]]
colors = [COLORS["accent2"], COLORS["success"]]

bars = ax.barh(labels, alphas, color=colors, height=0.5, zorder=3)
ax.axvline(0.7, color=COLORS["warning"], linestyle="--", linewidth=1.8, zorder=2)
ax.text(0.7, 1.65, "required threshold\n(Krippendorff's α ≥ 0.7)", color=COLORS["warning"],
        fontsize=9.5, ha="center", va="bottom")

for bar, alpha in zip(bars, alphas):
    ax.text(alpha + 0.02, bar.get_y() + bar.get_height() / 2, f"α = {alpha:.3f}",
            va="center", fontsize=12, fontweight="bold", color=COLORS["text"])

ax.set_xlim(0, 1.0)
style_axis(ax, grid=True)
ax.set_xlabel("Krippendorff's Alpha (inter-rater agreement)")
ax.set_title(f"Pilot Study: Inter-Rater Agreement  ({results['n_samples']} decisions, {results['n_pilot_raters']} pilot raters)",
             fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_agreement.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved {out_path}")
