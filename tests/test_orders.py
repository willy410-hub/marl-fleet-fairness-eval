import numpy as np
import pytest

from env.config import EnvConfig
from env.orders import OrderGenerator
from env.zones import build_zones


@pytest.fixture
def generator():
    config = EnvConfig()
    zones = build_zones(config.grid_size)
    rng = np.random.default_rng(42)
    return OrderGenerator(config, zones, rng), config


def test_pickup_never_equals_dropoff(generator):
    gen, config = generator
    for step in range(200):
        for order in gen.spawn_orders(step):
            assert order.pickup != order.dropoff


def test_payout_scales_with_distance(generator):
    gen, config = generator
    orders = []
    for step in range(300):
        orders.extend(gen.spawn_orders(step))
    assert len(orders) > 10, "need enough samples for a meaningful correlation check"

    distances = [o.distance for o in orders]
    payouts = [o.payout for o in orders]
    correlation = np.corrcoef(distances, payouts)[0, 1]
    assert correlation > 0.9, f"payout should be near-perfectly linear in distance, got corr={correlation}"


def test_generator_is_unbiased_over_a_long_run(generator):
    gen, config = generator
    n_steps = 24 * config.steps_per_hour * 20
    total = sum(len(gen.spawn_orders(step)) for step in range(n_steps))

    avg_multiplier = sum(config.demand_profile) / 24
    expected = config.order_spawn_rate * avg_multiplier * n_steps

    assert abs(total - expected) / expected < 0.05


def test_hourly_multiplier_cycles_correctly(generator):
    gen, config = generator
    assert gen.hourly_multiplier(0) == config.demand_profile[0]
    assert gen.hourly_multiplier(config.steps_per_hour) == config.demand_profile[1]
    assert gen.hourly_multiplier(25 * config.steps_per_hour) == config.demand_profile[1]
