"""
Runs a full episode against a given action-selection function,
collecting the raw per-step data needed to compute every metric in
evaluation/metrics.py.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from env.actions import ActionKind, decode_action
from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv

ActionFn = Callable[[FleetDispatchEnv, dict], dict[str, int]]


@dataclass
class EpisodeTrace:
    """Every raw signal collected from one episode, ready to feed into evaluation/metrics.py."""

    total_completed: int
    zone_completion_counts: np.ndarray
    zone_offer_counts: np.ndarray
    zone_rejection_counts: np.ndarray
    accepted_flags: list[int] = field(default_factory=list)
    offer_payouts: list[float] = field(default_factory=list)
    total_reward: float = 0.0


def run_episode(env: FleetDispatchEnv, action_fn: ActionFn, seed: int) -> EpisodeTrace:
    """
    Run one full episode using `action_fn(env, obs) -> actions` to
    choose actions each step, recording every accept/reject decision
    against the true payout of the offer involved (for
    DecisionQualityMetrics) and per-zone offer/rejection counts (for
    FairnessMetrics).
    """
    n_zones = len(env.zones)
    zone_offer_counts = np.zeros(n_zones, dtype=np.int64)
    zone_rejection_counts = np.zeros(n_zones, dtype=np.int64)
    accepted_flags: list[int] = []
    offer_payouts: list[float] = []
    total_reward = 0.0

    obs, infos = env.reset(seed=seed)

    while env.agents:
        actions = action_fn(env, obs)

        for aid, action_index in actions.items():
            driver = env.get_driver_states()[aid]
            visible_ids = env._last_offer_slots.get(aid, [])
            kind, param = decode_action(action_index, env.config)

            for oid in visible_ids:
                order = env._orders.get(oid)
                if order is None:
                    continue
                zone_offer_counts[order.zone_id] += 1

            if kind == ActionKind.ACCEPT_OFFER and param < len(visible_ids):
                oid = visible_ids[param]
                order = env._orders.get(oid)
                if order is not None and order.assigned_agent is None:
                    accepted_flags.append(1)
                    offer_payouts.append(order.payout)
            elif kind == ActionKind.REJECT_AND_IDLE and visible_ids:
                nearest_oid = min(
                    visible_ids,
                    key=lambda oid: driver.distance_to(env._orders[oid].pickup) if oid in env._orders else 1e9,
                )
                order = env._orders.get(nearest_oid)
                if order is not None:
                    accepted_flags.append(0)
                    offer_payouts.append(order.payout)
                    zone_rejection_counts[order.zone_id] += 1

        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += sum(rewards.values())

    driver_states = env.get_driver_states()
    total_completed = sum(d.completed_orders for d in driver_states.values())

    return EpisodeTrace(
        total_completed=total_completed,
        zone_completion_counts=env.get_zone_completion_counts(),
        zone_offer_counts=zone_offer_counts,
        zone_rejection_counts=zone_rejection_counts,
        accepted_flags=accepted_flags,
        offer_payouts=offer_payouts,
        total_reward=total_reward,
    )


def random_action_fn(env: FleetDispatchEnv, obs: dict) -> dict[str, int]:
    """A trivial action function for smoke-testing the evaluation pipeline without a trained policy."""
    rng = np.random.default_rng()
    return {aid: int(rng.integers(0, env.action_space(aid).n)) for aid in env.agents}
