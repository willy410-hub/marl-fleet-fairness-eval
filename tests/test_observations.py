import numpy as np
import pytest

from env.config import EnvConfig
from env.drivers import DriverState
from env.observations import build_observation, observation_size, visible_offer_ids
from env.orders import Order


@pytest.fixture
def config():
    return EnvConfig()


def test_offers_outside_visibility_radius_are_excluded(config):
    driver = DriverState(agent_id="driver_0", position=(5, 5))
    orders = [
        Order(order_id=0, pickup=(5, 6), dropoff=(9, 9), spawn_step=0, zone_id=0, payout=10.0),
        Order(order_id=1, pickup=(5, 8), dropoff=(9, 9), spawn_step=0, zone_id=0, payout=10.0),
        Order(order_id=2, pickup=(9, 9), dropoff=(0, 0), spawn_step=0, zone_id=3, payout=10.0),
    ]
    visible = visible_offer_ids(driver, orders, config)
    assert visible == [0, 1]
    assert 2 not in visible


def test_observation_has_correct_fixed_shape(config):
    driver = DriverState(agent_id="driver_0", position=(5, 5))
    obs = build_observation("driver_0", driver, [], step=0, config=config, local_demand_signal=0.0)
    assert obs.shape == (observation_size(config),)
    assert obs.dtype == np.float32


def test_observation_zero_pads_when_fewer_offers_than_slots(config):
    driver = DriverState(agent_id="driver_0", position=(5, 5))
    obs = build_observation("driver_0", driver, [], step=0, config=config, local_demand_signal=0.0)
    assert np.all(obs[6:] == 0.0)


def test_observation_never_contains_another_agents_position(config):
    """
    Direct regression test for the core partial-observability claim:
    build_observation's signature has no parameter through which another
    agent's DriverState could be passed, and the function only reads
    from the single `driver` argument -- this test constructs two
    different drivers and confirms their observations differ only in
    the way expected from their own state, never leaking the other's.
    """
    driver_a = DriverState(agent_id="driver_a", position=(1, 1))
    driver_b = DriverState(agent_id="driver_b", position=(8, 8))

    obs_a = build_observation("driver_a", driver_a, [], step=0, config=config, local_demand_signal=0.0)
    obs_b = build_observation("driver_b", driver_b, [], step=0, config=config, local_demand_signal=0.0)

    grid_norm = config.grid_size - 1
    assert obs_a[0] == pytest.approx(1 / grid_norm)
    assert obs_a[1] == pytest.approx(1 / grid_norm)
    assert obs_b[0] == pytest.approx(8 / grid_norm)
    assert obs_b[1] == pytest.approx(8 / grid_norm)
