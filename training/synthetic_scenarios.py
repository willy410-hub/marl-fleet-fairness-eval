"""
Automated synthetic training-scenario generation (Phase 2, addition 5).

`training/scenarios.py` defines exactly one hand-authored variant
(the surge scenario) used to demonstrate fine-tuning. This module
generalizes that into an automatic generator: instead of a human
hand-picking and hand-coding each new training variant, it produces a
batch of `n` distinct, reproducible synthetic scenarios by perturbing
a small, bounded set of scalar economic/demand-volume parameters --
the kind of automated data-generation step that lets a training
pipeline see far more scenario diversity per unit of human effort than
manually authoring each one would.

What is -- and is not -- perturbed, and why:

- `grid_size`, `n_zones`, and `n_agents` are NEVER touched. Changing
  any of them changes the observation/action space shape
  (`training/rllib_env_wrapper.py` derives both directly from these),
  which would silently break weight-compatibility with an existing
  trained policy -- exactly the kind of bug this module is designed
  to avoid, not introduce.
- The *shape* of `demand_profile` -- the real, TLC-calibrated hourly
  demand curve (see `data/tlc_calibration.py`) -- is never touched.
  Only its overall *volume* is scaled, using the same
  scale-the-real-curve pattern already used by
  `evaluation/robustness.py`'s demand-spike test and
  `training/scenarios.py`'s surge scenario. This module generalizes
  that one existing pattern into an automatic generator; it does not
  introduce a new, unrelated notion of "synthetic data."
- Every other perturbed field (`order_spawn_rate`,
  `base_payout_per_order`, `payout_distance_multiplier`,
  `fairness_weight`, `order_expiry_steps`) is a scalar economic or
  timing knob already exposed on `EnvConfig` -- perturbing it produces
  a fully valid `FleetDispatchEnv` instance, not a fabricated
  observation or reward.

This keeps the project's central, already-documented claim intact:
the environment's real-world calibration (demand shape, from real NYC
TLC trip data) is never synthetic. What *is* synthetic, and is labeled
as such everywhere it is used, is the additional scenario *diversity*
generated on top of that real baseline for training-time augmentation.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig

# Each perturbed parameter is scaled by a multiplier drawn uniformly
# from its (low, high) range below -- e.g. order_spawn_rate at 0.6
# means "as low as 60% of the baseline volume", at 1.6 means "as high
# as 160%". Ranges are deliberately bounded (never below ~50% or above
# ~160% of baseline) so every generated scenario stays a plausible
# variant of the real, calibrated environment -- not a degenerate edge
# case.
PERTURBATION_RANGES: dict[str, tuple[float, float]] = {
    "order_spawn_rate": (0.6, 1.6),
    "base_payout_per_order": (0.8, 1.3),
    "payout_distance_multiplier": (0.8, 1.3),
    "fairness_weight": (0.5, 1.5),
    "order_expiry_steps": (0.7, 1.5),
}


@dataclass(frozen=True)
class SyntheticScenario:
    """One generated training scenario: its config, and exactly what was perturbed to produce it."""

    scenario_id: str
    env_config: EnvConfig
    perturbation_multipliers: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "perturbation_multipliers": self.perturbation_multipliers,
            "env_config": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(self.env_config).items()
            },
        }


def generate_synthetic_scenario(
    base_config: EnvConfig, rng: np.random.Generator, scenario_id: str
) -> SyntheticScenario:
    """
    Produce one synthetic scenario by drawing an independent random
    multiplier for each field in PERTURBATION_RANGES and applying it
    to `base_config`. `order_expiry_steps` is rounded to a valid
    integer step count (minimum 1); every other perturbed field stays
    a float. `demand_profile`'s real hourly shape is preserved and
    only scaled by the same multiplier drawn for `order_spawn_rate`
    (see module docstring).
    """
    overrides: dict = {}
    multipliers: dict[str, float] = {}

    for param, (low, high) in PERTURBATION_RANGES.items():
        multiplier = float(rng.uniform(low, high))
        multipliers[param] = round(multiplier, 4)
        base_value = getattr(base_config, param)
        if param == "order_expiry_steps":
            overrides[param] = max(1, round(base_value * multiplier))
        else:
            overrides[param] = round(base_value * multiplier, 4)

    spawn_rate_multiplier = multipliers["order_spawn_rate"]
    overrides["demand_profile"] = tuple(
        round(hourly_value * spawn_rate_multiplier, 4) for hourly_value in base_config.demand_profile
    )

    new_config = dataclasses.replace(base_config, **overrides)
    return SyntheticScenario(scenario_id=scenario_id, env_config=new_config, perturbation_multipliers=multipliers)


def generate_synthetic_scenario_batch(
    base_config: EnvConfig, n_scenarios: int, seed: int
) -> list[SyntheticScenario]:
    """
    Generate `n_scenarios` distinct, reproducible synthetic scenarios
    from a single seed -- the same seed always reproduces the same
    batch, so a batch used for a training run can be regenerated
    exactly (or saved via `training/generate_synthetic_scenarios.py`)
    rather than depending on unrecorded randomness.
    """
    rng = np.random.default_rng(seed)
    return [
        generate_synthetic_scenario(base_config, rng, scenario_id=f"synthetic-{seed}-{i:03d}")
        for i in range(n_scenarios)
    ]
