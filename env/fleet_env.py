"""
FleetDispatchEnv: the PettingZoo ParallelEnv tying together zones,
orders, drivers, observations, actions, and rewards into one
step()-able multi-agent environment.

Implements PettingZoo's ParallelEnv API (all agents act simultaneously
each step), which is what RLlib's MultiAgentEnv wrapper expects (see
training/rllib_env_wrapper.py).
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from env.actions import ActionKind, action_space_size, decode_action
from env.config import EnvConfig
from env.drivers import DriverState
from env.fairness_metrics import zone_coverage_variance
from env.observations import build_observation, observation_size, visible_offer_ids
from env.orders import Order, OrderGenerator
from env.rewards import (
    RewardBreakdown,
    abandonment_penalty,
    cherry_picking_penalty,
    combine_reward,
    extrinsic_reward_on_delivery,
    intrinsic_fairness_reward,
    unsafe_shortcut_penalty,
)
from env.zones import Zone, build_zones, zone_of


class FleetDispatchEnv(ParallelEnv):
    """
    Multi-agent food-delivery fleet dispatch environment.

    Every driver-agent observes only its own state and nearby order
    offers (partial observability -- see env/observations.py), chooses
    each step whether to accept an offer, reject and idle, or move to
    a zone (optionally via the penalized "shortcut" variant), and is
    rewarded via the mixed extrinsic/fairness-intrinsic split defined
    in env/rewards.py.
    """

    metadata = {"name": "fleet_dispatch_v0", "render_modes": ["human", None]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.possible_agents = [f"driver_{i}" for i in range(self.config.n_agents)]
        self.agents = list(self.possible_agents)

        self.zones: list[Zone] = build_zones(self.config.grid_size)
        self._rng = np.random.default_rng(self.config.seed)

        # Populated in reset():
        self._drivers: dict[str, DriverState] = {}
        self._orders: dict[int, Order] = {}
        self._order_gen: OrderGenerator | None = None
        self._zone_completion_counts = np.zeros(len(self.zones), dtype=np.int64)
        self._step_count = 0
        self._last_offer_slots: dict[str, list[int]] = {}  # agent_id -> ordered visible order_ids
        self._last_reward_breakdown: dict[str, RewardBreakdown] = {}

    # ------------------------------------------------------------------
    # PettingZoo required space methods
    # ------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        size = observation_size(self.config)
        return Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return Discrete(action_space_size(self.config))

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)

        self.agents = list(self.possible_agents)
        self._order_gen = OrderGenerator(self.config, self.zones, self._rng)
        self._orders = {}
        self._zone_completion_counts = np.zeros(len(self.zones), dtype=np.int64)
        self._step_count = 0
        self._last_reward_breakdown = {}

        self._drivers = {
            agent_id: DriverState(
                agent_id=agent_id,
                position=(
                    int(self._rng.integers(0, self.config.grid_size)),
                    int(self._rng.integers(0, self.config.grid_size)),
                ),
            )
            for agent_id in self.agents
        }

        self._spawn_new_orders()
        observations = self._build_all_observations()
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        rewards: dict[str, float] = {}
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}
        infos: dict[str, dict] = {agent_id: {} for agent_id in self.agents}

        # Process actions in a fixed but arbitrary order (agent list order).
        # Orders are claimed first-come-first-served within a step -- an
        # order accepted by one agent is immediately removed from
        # `self._orders` so a later agent in the same step cannot also
        # claim it (see _apply_action's guard on order.assigned_agent).
        for agent_id in self.agents:
            action_index = actions.get(agent_id)
            if action_index is None:
                rewards[agent_id] = 0.0
                continue
            breakdown = self._apply_action(agent_id, action_index)
            self._last_reward_breakdown[agent_id] = breakdown
            rewards[agent_id] = breakdown.total
            infos[agent_id]["reward_breakdown"] = breakdown

        self._advance_active_deliveries()
        self._expire_stale_orders()
        self._spawn_new_orders()

        self._step_count += 1
        episode_done = self._step_count >= self.config.steps_per_episode
        if episode_done:
            truncations = {agent_id: True for agent_id in self.agents}

        observations = self._build_all_observations()

        if episode_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _apply_action(self, agent_id: str, action_index: int) -> RewardBreakdown:
        driver = self._drivers[agent_id]
        kind, param = decode_action(action_index, self.config)

        extrinsic = 0.0
        cherry_penalty = 0.0
        shortcut_penalty = 0.0

        visible_ids = self._last_offer_slots.get(agent_id, [])

        if kind == ActionKind.ACCEPT_OFFER:
            extrinsic = self._try_accept_offer(driver, visible_ids, param)

        elif kind == ActionKind.REJECT_AND_IDLE:
            cherry_penalty = self._maybe_penalize_rejection(driver, visible_ids)
            driver.steps_idle += 1

        elif kind == ActionKind.MOVE_TO_ZONE:
            self._move_driver_toward_zone(driver, param, risky=False)

        elif kind == ActionKind.MOVE_TO_ZONE_SHORTCUT:
            self._move_driver_toward_zone(driver, param, risky=True)
            shortcut_penalty = unsafe_shortcut_penalty(True, self.config)

        intrinsic = intrinsic_fairness_reward(self._zone_completion_counts, self.config)
        combined_extrinsic_intrinsic = combine_reward(extrinsic, intrinsic, self.config)
        abandon_penalty = abandonment_penalty(self._zone_completion_counts, self.config)

        return RewardBreakdown(
            extrinsic=combined_extrinsic_intrinsic,
            intrinsic_fairness=0.0,
            cherry_picking_penalty=cherry_penalty,
            abandonment_penalty=abandon_penalty,
            unsafe_shortcut_penalty=shortcut_penalty,
        )

    def _try_accept_offer(self, driver: DriverState, visible_ids: list[int], slot: int) -> float:
        if slot >= len(visible_ids):
            return 0.0
        order_id = visible_ids[slot]
        order = self._orders.get(order_id)
        if order is None or order.assigned_agent is not None:
            return 0.0
        if len(driver.active_order_ids) >= self.config.max_active_orders_per_agent:
            return 0.0

        order.assigned_agent = driver.agent_id
        order.accepted_step = self._step_count
        driver.active_order_ids.append(order.order_id)
        driver.steps_idle = 0
        return 0.0

    def _maybe_penalize_rejection(self, driver: DriverState, visible_ids: list[int]) -> float:
        if not visible_ids:
            return 0.0
        visible_orders = [self._orders[oid] for oid in visible_ids if oid in self._orders]
        if not visible_orders:
            return 0.0
        mean_payout = float(np.mean([o.payout for o in visible_orders]))
        nearest = min(visible_orders, key=lambda o: driver.distance_to(o.pickup))
        return cherry_picking_penalty(
            rejected_payout=nearest.payout,
            mean_visible_payout=mean_payout,
            was_idle=driver.is_available,
            config=self.config,
        )

    def _move_driver_toward_zone(self, driver: DriverState, target_zone_id: int, risky: bool) -> None:
        target_zone = self.zones[target_zone_id]
        target_center = target_zone.center
        row, col = driver.position

        step_size = 2 if risky else 1

        new_row = row + step_size * np.sign(target_center[0] - row)
        new_col = col + step_size * np.sign(target_center[1] - col)
        new_row = int(np.clip(new_row, 0, self.config.grid_size - 1))
        new_col = int(np.clip(new_col, 0, self.config.grid_size - 1))
        driver.position = (new_row, new_col)

    # ------------------------------------------------------------------
    # World update
    # ------------------------------------------------------------------
    def _advance_active_deliveries(self) -> None:
        for driver in self._drivers.values():
            if not driver.active_order_ids:
                continue

            order_id = driver.active_order_ids[0]
            order = self._orders.get(order_id)
            if order is None:
                driver.active_order_ids.remove(order_id)
                continue

            target = order.dropoff if order.picked_up else order.pickup
            row, col = driver.position
            new_row = row + int(np.sign(target[0] - row))
            new_col = col + int(np.sign(target[1] - col))
            driver.position = (new_row, new_col)

            if driver.position == target:
                if not order.picked_up:
                    order.picked_up = True
                else:
                    self._complete_order(driver, order)

    def _complete_order(self, driver: DriverState, order: Order) -> None:
        delivery_time = self._step_count - order.accepted_step
        extrinsic = extrinsic_reward_on_delivery(
            payout=order.payout,
            delivery_time_steps=max(delivery_time, 0),
            distance_traveled=order.distance,
            config=self.config,
        )
        driver.total_earnings += extrinsic
        driver.completed_orders += 1
        driver.active_order_ids.remove(order.order_id)
        self._zone_completion_counts[order.zone_id] += 1
        del self._orders[order.order_id]

    def _expire_stale_orders(self) -> None:
        expired_ids = [
            oid
            for oid, order in self._orders.items()
            if order.is_expired(self._step_count, self.config.order_expiry_steps)
        ]
        for oid in expired_ids:
            del self._orders[oid]

    def _spawn_new_orders(self) -> None:
        for order in self._order_gen.spawn_orders(self._step_count):
            self._orders[order.order_id] = order

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _build_all_observations(self) -> dict[str, np.ndarray]:
        observations = {}
        open_orders = [o for o in self._orders.values() if o.assigned_agent is None]

        for agent_id in self.agents:
            driver = self._drivers[agent_id]
            visible_ids = visible_offer_ids(driver, open_orders, self.config)
            self._last_offer_slots[agent_id] = visible_ids

            visible = [self._orders[oid] for oid in visible_ids]
            zid = zone_of(driver.position[0], driver.position[1], self.zones)
            local_demand = self._local_demand_signal(zid)

            observations[agent_id] = build_observation(
                agent_id=agent_id,
                driver=driver,
                visible_offers=visible,
                step=self._step_count,
                config=self.config,
                local_demand_signal=local_demand,
            )
        return observations

    def _local_demand_signal(self, zone_id: int) -> float:
        open_orders = [o for o in self._orders.values() if o.assigned_agent is None]
        if not open_orders:
            return 0.0
        in_zone = sum(1 for o in open_orders if o.zone_id == zone_id)
        return in_zone / len(open_orders)

    # ------------------------------------------------------------------
    # Introspection helpers used by evaluation/ and annotation_app/
    # ------------------------------------------------------------------
    def get_zone_completion_counts(self) -> np.ndarray:
        return self._zone_completion_counts.copy()

    def get_driver_states(self) -> dict[str, DriverState]:
        return dict(self._drivers)

    def render(self):
        if self.render_mode != "human":
            return
        grid = [["." for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)]
        for driver in self._drivers.values():
            r, c = driver.position
            grid[r][c] = "D"
        print(f"--- step {self._step_count} ---")
        for row in grid:
            print(" ".join(row))
