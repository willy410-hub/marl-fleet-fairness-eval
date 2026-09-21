"""
Named scenario configs used by the fine-tuning pipeline (Phase 2,
addition 2) and the benchmark package (Phase 2, addition 4).

The baseline PPO policy (`training/train_ppo.py`) is trained under
`EnvConfig()`'s default order-spawn rate, which is itself the real,
TLC-calibrated *shape* of demand across the day (see
`data/tlc_calibration.py`) but at a moderate overall *volume*
(`order_spawn_rate=0.35`). This module defines a second, harder
scenario -- a sustained citywide demand surge -- that the baseline
policy was never trained on, so continuing training on it is a
genuine distribution shift, not a relabeled copy of the same config.

This mirrors the same `dataclasses.replace`-based pattern already used
for the robustness stress test in `evaluation/robustness.py`, applied
here to define a *training* scenario rather than a one-off eval
condition.
"""
import dataclasses

from env.config import EnvConfig

# A sustained ~70% increase in order arrival rate, applied uniformly on
# top of the existing (already realistic, TLC-shaped) hourly demand
# curve -- e.g. a new market with materially higher order density, or
# a permanent step-change in platform adoption, rather than a
# transient rush-hour spike (which `evaluation/robustness.py` already
# covers as a no-retraining stress test).
SURGE_ORDER_SPAWN_RATE = 0.60


def build_surge_scenario_config(base_config: EnvConfig) -> EnvConfig:
    """
    Return a new EnvConfig representing a sustained higher-demand
    market: a genuinely different training distribution from the
    baseline, used to demonstrate continued training / fine-tuning of
    an existing checkpoint rather than training a fresh policy from
    scratch.
    """
    return dataclasses.replace(base_config, order_spawn_rate=SURGE_ORDER_SPAWN_RATE)
