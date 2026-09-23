"""
Rule-based safety shield (Phase 2, addition 6): explicit, machine-checkable
hard-constraint enforcement, layered on top of the environment's existing
reward-shaped (soft) constraints.

Two constraints are made explicit here, and they are deliberately
different in kind, because interview feedback on this project
specifically asked for the *difference* between "the reward balances
competing objectives" and "a rule enforces a hard limit" to be spelled
out concretely:

1. ACCEPT_OFFER legality (capacity + ownership). FleetDispatchEnv
   already prevents these from having any effect --
   `_try_accept_offer` in env/fleet_env.py silently no-ops an
   accept of an already-claimed order, or one that would push a
   driver over `max_active_orders_per_agent`. That enforcement is
   correct but implicit: nothing exposes it as a checkable mask ahead
   of time, so neither a training-time action-masking model nor an
   evaluator auditing a policy's behavior can see *which* actions
   were legal before one was chosen. `compute_action_mask` below makes
   that existing invariant explicit and inspectable without changing
   the invariant itself.

2. The "unsafe shortcut" move (MOVE_TO_ZONE_SHORTCUT). This is a
   genuinely different case: today it is a *soft* constraint --
   env/rewards.py's `unsafe_shortcut_penalty` always lets the action
   through and only makes it costly. There is no way, today, to
   actually forbid it. `compute_action_mask(..., disallow_shortcuts=True)`
   turns that soft, reward-shaped discouragement into a hard,
   rule-enforced constraint: when enabled, the shortcut actions are
   masked illegal outright, and `ShieldedActionFn` guarantees zero of
   them ever reach the environment, regardless of what the wrapped
   policy proposes -- the concrete "rule-based safety shield ...
   incorporated into training and deployment pipelines" pattern.

What this module does NOT claim to be: a CMDP/Lagrangian
constrained-optimization method. That is a different, complementary
technique (learning a dual multiplier that dynamically re-weights a
soft constraint's penalty until an expected-violation target is met,
rather than a fixed-weight reward term or an outright rule). This
module implements the rule-based-shield half only; the Lagrangian
half is documented as a natural extension in the README, not
implemented here, so as not to overstate what exists in the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from env.actions import ActionKind, action_space_size, decode_action
from env.config import EnvConfig
from env.drivers import DriverState
from env.orders import Order


def reject_and_idle_action_index(config: EnvConfig) -> int:
    """
    The single, always-legal fallback action index: REJECT_AND_IDLE.
    Derived from action_space_size's own layout (env/actions.py) rather
    than hard-coded, so it can never silently drift out of sync with a
    change to the action-space layout.
    """
    return config.max_open_offers_per_agent


def compute_action_mask(
    driver: DriverState,
    visible_ids: list[int],
    orders: dict[int, Order],
    config: EnvConfig,
    disallow_shortcuts: bool = False,
) -> np.ndarray:
    """
    Return a boolean array of length `action_space_size(config)`:
    True = legal to take right now, False = the shield will not let it
    through unmodified.

    - ACCEPT_OFFER(slot): illegal if there is no offer in that slot,
      the referenced order no longer exists or is already claimed by
      another driver, or `driver` is already at
      `config.max_active_orders_per_agent` (mirrors
      FleetDispatchEnv._try_accept_offer's own no-op guards).
    - REJECT_AND_IDLE and MOVE_TO_ZONE(*): always legal.
    - MOVE_TO_ZONE_SHORTCUT(*): legal unless `disallow_shortcuts=True`,
      in which case every shortcut action is masked illegal -- the
      hard-constraint mode described in the module docstring.
    """
    n_actions = action_space_size(config)
    mask = np.ones(n_actions, dtype=bool)
    at_capacity = len(driver.active_order_ids) >= config.max_active_orders_per_agent

    for action_index in range(n_actions):
        kind, param = decode_action(action_index, config)

        if kind == ActionKind.ACCEPT_OFFER:
            if param >= len(visible_ids):
                mask[action_index] = False
                continue
            order = orders.get(visible_ids[param])
            if order is None or order.assigned_agent is not None or at_capacity:
                mask[action_index] = False

        elif kind == ActionKind.MOVE_TO_ZONE_SHORTCUT and disallow_shortcuts:
            mask[action_index] = False

    return mask


@dataclass
class ShieldDecision:
    """One shield decision for one agent in one step."""

    agent_id: str
    proposed_action: int
    final_action: int
    was_overridden: bool


def shield_action(action_index: int, mask: np.ndarray, config: EnvConfig) -> tuple[int, bool]:
    """
    If `action_index` is legal per `mask`, pass it through unchanged.
    Otherwise substitute the always-legal REJECT_AND_IDLE action.
    Returns (final_action_index, was_overridden).
    """
    if mask[action_index]:
        return action_index, False
    return reject_and_idle_action_index(config), True


class ShieldedActionFn:
    """
    Wraps an existing evaluation/episode_runner.ActionFn so every
    action it proposes is checked against `compute_action_mask` before
    it reaches the environment. Any illegal or (optionally) forbidden
    action is transparently replaced with REJECT_AND_IDLE, and a
    per-episode override count is recorded on `self.override_log` for
    auditing -- e.g. "the unshielded policy proposed 12 shortcut moves
    this episode; the shield blocked all 12."

    Usable as a drop-in ActionFn with evaluation.episode_runner.run_episode,
    evaluation.report.evaluate_policy, and benchmark.run_benchmark --
    a shielded policy can be benchmarked exactly like an unshielded one.
    """

    def __init__(self, inner_action_fn: Callable, disallow_shortcuts: bool = False):
        self.inner_action_fn = inner_action_fn
        self.disallow_shortcuts = disallow_shortcuts
        self.override_log: list[ShieldDecision] = []

    def __call__(self, env, obs: dict) -> dict[str, int]:
        proposed = self.inner_action_fn(env, obs)
        final_actions: dict[str, int] = {}

        for agent_id, action_index in proposed.items():
            driver = env.get_driver_states()[agent_id]
            visible_ids = env._last_offer_slots.get(agent_id, [])
            mask = compute_action_mask(
                driver, visible_ids, env._orders, env.config, disallow_shortcuts=self.disallow_shortcuts
            )
            final_action, was_overridden = shield_action(action_index, mask, env.config)
            final_actions[agent_id] = final_action
            if was_overridden:
                self.override_log.append(
                    ShieldDecision(
                        agent_id=agent_id,
                        proposed_action=action_index,
                        final_action=final_action,
                        was_overridden=True,
                    )
                )

        return final_actions

    @property
    def override_count(self) -> int:
        return len(self.override_log)

    def reset_log(self) -> None:
        self.override_log = []
