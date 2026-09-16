"""
Configuration for the FleetDispatchEnv multi-agent environment.

Centralizing every tunable constant here (grid size, reward weights,
episode length, etc.) means the environment, the training scripts, and
the evaluation scripts all read from one source of truth instead of
scattering magic numbers throughout the codebase.
"""
from dataclasses import dataclass, field

from data.tlc_calibration import build_calibrated_demand_profile


@dataclass(frozen=True)
class EnvConfig:
    """Immutable configuration for one FleetDispatchEnv instance."""

    # --- City grid ---
    grid_size: int = 10               # city is a grid_size x grid_size grid of cells
    n_zones: int = 4                  # grid is partitioned into n_zones quadrant zones

    # --- Fleet ---
    n_agents: int = 8                 # number of driver-agents
    max_active_orders_per_agent: int = 3

    # --- Orders ---
    order_spawn_rate: float = 0.35    # expected new orders per step (Poisson lambda)
    max_open_offers_per_agent: int = 4  # how many nearby offers an agent can see at once
    offer_visibility_radius: int = 3  # Chebyshev distance within which offers are visible
    order_expiry_steps: int = 6       # an unaccepted offer disappears after this many steps

    # --- Time ---
    steps_per_episode: int = 200
    steps_per_hour: int = 12          # used to derive a synthetic time-of-day signal

    # --- Economics (extrinsic reward) ---
    base_payout_per_order: float = 8.0
    payout_distance_multiplier: float = 1.2   # extra payout per grid cell of order distance
    time_penalty_per_step: float = 0.15       # r_ext penalty per step an order is in progress
    fuel_cost_per_cell: float = 0.3           # r_ext penalty per grid cell traveled

    # --- Fairness (intrinsic reward) ---
    fairness_weight: float = 0.6      # omega in r = r_ext + omega * r_int
    coverage_bonus_scale: float = 5.0 # scales (1 - zone_coverage_variance) into a reward

    # --- Penalties (judged decision quality) ---
    cherry_picking_penalty: float = 1.5   # penalty for rejecting a low-value offer while idle nearby
    abandonment_penalty: float = 2.0      # penalty applied to zone coverage when a low-demand zone is starved
    unsafe_shortcut_penalty: float = 3.0  # penalty for the risk-taking action variant (see env/actions.py)

    # --- Demand shaping (calibrated from real NYC TLC trip data --
    # see data/tlc_calibration.py for the extraction methodology,
    # source data, and an explicit note on where this profile is a
    # direct measurement vs. a documented, cited adjustment) ---
    demand_profile: tuple = field(default_factory=build_calibrated_demand_profile)

    # --- Random seed ---
    seed: int | None = None
