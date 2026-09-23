import numpy as np
import pytest

from env.actions import ActionKind, action_space_size, decode_action
from env.config import EnvConfig
from env.drivers import DriverState
from env.fleet_env import FleetDispatchEnv
from env.orders import Order
from env.safety_shield import (
    ShieldedActionFn,
    compute_action_mask,
    reject_and_idle_action_index,
    shield_action,
)


@pytest.fixture
def config():
    return EnvConfig(seed=1, max_open_offers_per_agent=4, n_zones=4, max_active_orders_per_agent=2)


@pytest.fixture
def driver():
    return DriverState(agent_id="driver_0", position=(0, 0))


def _accept_index(slot: int) -> int:
    return slot  # ACCEPT_OFFER slots occupy the first max_open_offers_per_agent indices


def test_reject_index_is_always_legal_and_matches_decode(config):
    idx = reject_and_idle_action_index(config)
    kind, _ = decode_action(idx, config)
    assert kind == ActionKind.REJECT_AND_IDLE


def test_accept_legal_offer_is_unmasked(config, driver):
    order = Order(order_id=1, pickup=(0, 0), dropoff=(1, 1), spawn_step=0, zone_id=0, payout=10.0)
    mask = compute_action_mask(driver, visible_ids=[1], orders={1: order}, config=config)
    assert mask[_accept_index(0)]


def test_accept_empty_slot_is_illegal(config, driver):
    mask = compute_action_mask(driver, visible_ids=[], orders={}, config=config)
    assert not mask[_accept_index(0)]


def test_accept_already_claimed_order_is_illegal(config, driver):
    order = Order(order_id=1, pickup=(0, 0), dropoff=(1, 1), spawn_step=0, zone_id=0, payout=10.0, assigned_agent="driver_9")
    mask = compute_action_mask(driver, visible_ids=[1], orders={1: order}, config=config)
    assert not mask[_accept_index(0)]


def test_accept_at_capacity_is_illegal_for_every_slot(config):
    driver = DriverState(agent_id="driver_0", position=(0, 0), active_order_ids=[100, 101])  # at capacity (2)
    orders = {i: Order(order_id=i, pickup=(0, 0), dropoff=(1, 1), spawn_step=0, zone_id=0, payout=10.0) for i in range(4)}
    mask = compute_action_mask(driver, visible_ids=list(orders), orders=orders, config=config)
    n_accept = config.max_open_offers_per_agent
    assert not mask[:n_accept].any()


def test_non_accept_actions_always_legal_by_default(config, driver):
    mask = compute_action_mask(driver, visible_ids=[], orders={}, config=config)
    n_accept = config.max_open_offers_per_agent
    assert mask[n_accept:].all()


def test_disallow_shortcuts_masks_every_shortcut_action(config, driver):
    mask = compute_action_mask(driver, visible_ids=[], orders={}, config=config, disallow_shortcuts=True)
    for action_index in range(action_space_size(config)):
        kind, _ = decode_action(action_index, config)
        if kind == ActionKind.MOVE_TO_ZONE_SHORTCUT:
            assert not mask[action_index]
        elif kind == ActionKind.MOVE_TO_ZONE:
            assert mask[action_index]  # the safe variant stays legal


def test_shield_action_passes_through_legal_action(config):
    mask = np.ones(action_space_size(config), dtype=bool)
    final, overridden = shield_action(0, mask, config)
    assert final == 0
    assert overridden is False


def test_shield_action_substitutes_reject_for_illegal_action(config):
    mask = np.ones(action_space_size(config), dtype=bool)
    mask[0] = False
    final, overridden = shield_action(0, mask, config)
    assert overridden is True
    assert final == reject_and_idle_action_index(config)
    kind, _ = decode_action(final, config)
    assert kind == ActionKind.REJECT_AND_IDLE


def test_shielded_action_fn_blocks_every_shortcut_over_a_rollout():
    """
    An inner policy that always tries the shortcut move must never
    actually get a shortcut move applied once wrapped in
    ShieldedActionFn(disallow_shortcuts=True) -- the concrete
    hard-constraint guarantee this module exists to provide.
    """
    config = EnvConfig(seed=3, steps_per_episode=15)
    env = FleetDispatchEnv(config)
    obs, infos = env.reset(seed=3)

    def always_shortcut_fn(env, obs):
        # MOVE_TO_ZONE_SHORTCUT for zone 0 is always a valid action index.
        n_accept = env.config.max_open_offers_per_agent
        shortcut_index = n_accept + 1 + env.config.n_zones  # first shortcut action
        return {aid: shortcut_index for aid in env.agents}

    shield = ShieldedActionFn(always_shortcut_fn, disallow_shortcuts=True)

    while env.agents:
        actions = shield(env, obs)
        for aid, action_index in actions.items():
            kind, _ = decode_action(action_index, env.config)
            assert kind != ActionKind.MOVE_TO_ZONE_SHORTCUT
        obs, rewards, terms, truncs, infos = env.step(actions)

    assert shield.override_count > 0  # the shield actually had to intervene, not a vacuous pass


def test_unshielded_same_policy_does_take_shortcuts():
    """Sanity check that the previous test's guarantee comes from the shield, not from the
    environment already forbidding shortcuts on its own."""
    config = EnvConfig(seed=3, steps_per_episode=15)
    env = FleetDispatchEnv(config)
    obs, infos = env.reset(seed=3)

    n_accept = config.max_open_offers_per_agent
    shortcut_index = n_accept + 1 + config.n_zones
    took_shortcut = False

    for _ in range(config.steps_per_episode):
        actions = {aid: shortcut_index for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for aid, breakdown in ((a, i.get("reward_breakdown")) for a, i in infos.items() if a != "__all__"):
            if breakdown is not None and breakdown.unsafe_shortcut_penalty != 0.0:
                took_shortcut = True
        if not env.agents:
            break

    assert took_shortcut
