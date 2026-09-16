"""Generates assets/diagrams/demand_profile.png -- the real-data-calibrated hourly demand curve."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
from _style import COLORS, style_axis
from data.tlc_calibration import REAL_TLC_HOURLY_MULTIPLIERS, LUNCH_BUMP_HOURS, build_calibrated_demand_profile

fig, ax = plt.subplots(figsize=(11, 5.5))

hours = list(range(24))
raw_tlc = list(REAL_TLC_HOURLY_MULTIPLIERS)
calibrated = list(build_calibrated_demand_profile())

ax.plot(hours, raw_tlc, color=COLORS["text_dim"], linewidth=2, linestyle="--", marker="o", markersize=4,
        label="Real NYC TLC hourly trip volume (measured, Jan 2019)")
ax.plot(hours, calibrated, color=COLORS["accent"], linewidth=2.5, marker="o", markersize=5,
        label="agentctl demand_profile (TLC shape + documented lunch adjustment)")

for h in LUNCH_BUMP_HOURS:
    ax.axvspan(h - 0.5, h + 0.5, color=COLORS["accent3"], alpha=0.12, zorder=0)
ax.text(12, 0.15, "documented lunch-hour\nadjustment (×1.15)", ha="center", fontsize=8.5, color=COLORS["accent3"])

style_axis(ax)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Demand multiplier (mean = 1.0)")
ax.set_title("Order-Spawn Demand Profile: Calibrated From Real Trip Data", fontsize=13.5, fontweight="bold", pad=12)
ax.set_xticks(range(0, 24, 2))
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.set_ylim(0, 2.0)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demand_profile.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved {out_path}")
