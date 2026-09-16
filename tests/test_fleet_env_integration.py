import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv


def test_pettingzoo_official_api_compliance():
    """The environment must pass PettingZoo's own compliance suite."""
    env = FleetDispatchEnv(EnvConfig(seed=1, steps_per_episode=50))
    parallel_api_test(env, num_cycles=45)


def test_full_episode_runs_without_crashing():
    config = EnvConfig(seed=42, steps_per_episode=100)
    env = FleetDispatchEnv(config)
    obs, infos = env.reset(seed=42)

    rng = np.random.default_rng(0)
    steps = 0
    while env.agents:
        actions = {aid: int(rng.integers(0, env.action_space(aid).n)) for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1

    assert steps == config.steps_per_episode


def test_no_order_is_ever_double_assigned_under_maximum_contention():
    """
    Stress test: force every agent to compete for the same offer slot
    every step. If the first-come-first-served claiming guard in
    FleetDispatchEnv._try_accept_offer is broken, an order_id would
    appear active in more than one driver's active_order_ids at once.
    """
    config = EnvConfig(seed=7, n_agents=8, steps_per_episode=100)
    env = FleetDispatchEnv(config)
    env.reset(seed=7)

    while env.agents:
        actions = {aid: 0 for aid in env.agents}
        env.step(actions)

        all_active = []
        for driver in env.get_driver_states().values():
            all_active.extend(driver.active_order_ids)
        assert len(all_active) == len(set(all_active)), "an order was claimed by more than one driver at once"


def test_completion_counts_are_internally_consistent():
    config = EnvConfig(seed=7, n_agents=8, steps_per_episode=100)
    env = FleetDispatchEnv(config)
    env.reset(seed=7)

    while env.agents:
        actions = {aid: 0 for aid in env.agents}
        env.step(actions)

    total_from_drivers = sum(d.completed_orders for d in env.get_driver_states().values())
    total_from_zones = env.get_zone_completion_counts().sum()
    assert total_from_drivers == total_from_zones


def test_partial_observability_holds_throughout_a_live_episode():
    """
    Direct instrumented check across a running episode (not just a
    static unit test): at every step, every offer visible to every
    agent must be within that agent's configured visibility radius.
    """
    config = EnvConfig(seed=11, offer_visibility_radius=3, steps_per_episode=50)
    env = FleetDispatchEnv(config)
    env.reset(seed=11)

    violations = 0
    while env.agents:
        actions = {aid: 0 for aid in env.agents}
        env.step(actions)
        if not env.agents:
            break
        for aid in env.agents:
            driver = env.get_driver_states()[aid]
            for oid in env._last_offer_slots.get(aid, []):
                order = env._orders.get(oid)
                if order is None:
                    continue
                if driver.distance_to(order.pickup) > config.offer_visibility_radius:
                    violations += 1

    assert violations == 0


def test_all_agents_always_rejecting_never_completes_any_order():
    config = EnvConfig(seed=3, steps_per_episode=100)
    env = FleetDispatchEnv(config)
    env.reset(seed=3)

    reject_action = config.max_open_offers_per_agent
    while env.agents:
        actions = {aid: reject_action for aid in env.agents}
        env.step(actions)

    assert env.get_zone_completion_counts().sum() == 0
