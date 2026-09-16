"""
Partial-observability observation construction.

This is the module that enforces the design document's core
constraint: "each driver-agent sees own location + nearby order
offers, NOT full city order queue or other drivers' state." Every
observation returned here is deliberately scoped to what one agent
would realistically know in a real dispatch app -- nothing here reads
another agent's position or the full order backlog.
"""
import numpy as np

from env.config import EnvConfig
from env.drivers import DriverState
from env.orders import Order

# Observation layout (fixed-size vector, required by PettingZoo/RLlib's
# Box space): each offer slot contributes 4 floats (dx, dy, payout, distance),
# padded with zeros when fewer than max_open_offers_per_agent offers are visible.
OFFER_FIELDS = 4


def observation_size(config: EnvConfig) -> int:
    """Total length of the flat observation vector for one agent."""
    own_state_fields = 4          # normalized (row, col, active_orders_frac, earnings_norm)
    context_fields = 2            # normalized (time_of_day, local_demand_signal)
    offer_fields = config.max_open_offers_per_agent * OFFER_FIELDS
    return own_state_fields + context_fields + offer_fields


def build_observation(
    agent_id: str,
    driver: DriverState,
    visible_offers: list[Order],
    step: int,
    config: EnvConfig,
    local_demand_signal: float,
) -> np.ndarray:
    """
    Build one agent's observation vector.

    `visible_offers` must already be filtered to only the offers within
    this agent's offer_visibility_radius (done by env/observations.py's
    caller in env/fleet_env.py) -- this function does not do that
    filtering itself, so it cannot accidentally leak orders outside the
    agent's visibility radius.
    """
    grid_norm = config.grid_size - 1

    own_row = driver.position[0] / grid_norm
    own_col = driver.position[1] / grid_norm
    active_frac = len(driver.active_order_ids) / config.max_active_orders_per_agent
    earnings_norm = np.tanh(driver.total_earnings / 100.0)  # bounded, monotonic squashing

    time_of_day = ((step // config.steps_per_hour) % 24) / 24.0

    own_state = [own_row, own_col, active_frac, earnings_norm]
    context = [time_of_day, local_demand_signal]

    offer_vec: list[float] = []
    sorted_offers = sorted(visible_offers, key=lambda o: driver.distance_to(o.pickup))
    for offer in sorted_offers[: config.max_open_offers_per_agent]:
        dx = (offer.pickup[0] - driver.position[0]) / grid_norm
        dy = (offer.pickup[1] - driver.position[1]) / grid_norm
        payout_norm = np.tanh(offer.payout / 30.0)
        distance_norm = driver.distance_to(offer.pickup) / grid_norm
        offer_vec.extend([dx, dy, payout_norm, distance_norm])

    # Pad with zeros if fewer offers are visible than the fixed slot count.
    missing_slots = config.max_open_offers_per_agent - len(sorted_offers[: config.max_open_offers_per_agent])
    offer_vec.extend([0.0] * (missing_slots * OFFER_FIELDS))

    obs = np.array(own_state + context + offer_vec, dtype=np.float32)
    expected_len = observation_size(config)
    assert obs.shape == (expected_len,), f"Observation length mismatch: {obs.shape} vs expected ({expected_len},)"
    return obs


def visible_offer_ids(
    driver: DriverState, all_open_orders: list[Order], config: EnvConfig
) -> list[int]:
    """
    Return the order_ids of orders within this driver's visibility
    radius. This is the single choke point that enforces partial
    observability for offers -- env/fleet_env.py must route through
    this function rather than handing an agent the full order list.
    """
    return [
        order.order_id
        for order in all_open_orders
        if driver.distance_to(order.pickup) <= config.offer_visibility_radius
    ]
