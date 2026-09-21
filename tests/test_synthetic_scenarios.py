import dataclasses

import numpy as np
import pytest

from env.config import EnvConfig
from training.synthetic_scenarios import (
    PERTURBATION_RANGES,
    generate_synthetic_scenario,
    generate_synthetic_scenario_batch,
)


@pytest.fixture
def base_config():
    return EnvConfig(seed=1, steps_per_episode=60)


def test_batch_generates_requested_count(base_config):
    batch = generate_synthetic_scenario_batch(base_config, n_scenarios=10, seed=42)
    assert len(batch) == 10


def test_batch_scenario_ids_are_unique(base_config):
    batch = generate_synthetic_scenario_batch(base_config, n_scenarios=10, seed=42)
    ids = [s.scenario_id for s in batch]
    assert len(ids) == len(set(ids))


def test_same_seed_reproduces_identical_batch(base_config):
    """A saved seed must regenerate the exact same scenarios -- no unrecorded randomness."""
    batch_a = generate_synthetic_scenario_batch(base_config, n_scenarios=8, seed=123)
    batch_b = generate_synthetic_scenario_batch(base_config, n_scenarios=8, seed=123)
    for a, b in zip(batch_a, batch_b):
        assert a.env_config == b.env_config
        assert a.perturbation_multipliers == b.perturbation_multipliers


def test_different_seeds_produce_different_batches(base_config):
    batch_a = generate_synthetic_scenario_batch(base_config, n_scenarios=5, seed=1)
    batch_b = generate_synthetic_scenario_batch(base_config, n_scenarios=5, seed=2)
    assert any(a.env_config != b.env_config for a, b in zip(batch_a, batch_b))


def test_perturbed_fields_stay_within_documented_bounds(base_config):
    rng = np.random.default_rng(7)
    for i in range(200):
        scenario = generate_synthetic_scenario(base_config, rng, scenario_id=f"s{i}")
        for param, (low, high) in PERTURBATION_RANGES.items():
            multiplier = scenario.perturbation_multipliers[param]
            assert low <= multiplier <= high, f"{param} multiplier {multiplier} outside [{low}, {high}]"


def test_structural_fields_never_perturbed(base_config):
    """
    grid_size, n_zones, and n_agents must be byte-for-byte identical to
    the base config in every generated scenario -- changing any of
    them would change the observation/action space shape and silently
    break weight-transfer compatibility with an existing policy.
    """
    batch = generate_synthetic_scenario_batch(base_config, n_scenarios=20, seed=99)
    for scenario in batch:
        assert scenario.env_config.grid_size == base_config.grid_size
        assert scenario.env_config.n_zones == base_config.n_zones
        assert scenario.env_config.n_agents == base_config.n_agents
        assert scenario.env_config.max_active_orders_per_agent == base_config.max_active_orders_per_agent
        assert scenario.env_config.offer_visibility_radius == base_config.offer_visibility_radius


def test_demand_profile_shape_preserved_only_volume_scaled(base_config):
    """
    The real, TLC-calibrated demand curve's *shape* must be preserved:
    every generated scenario's demand_profile should equal the base
    profile scaled by a single constant (the order_spawn_rate
    multiplier), not an independently reshaped curve.
    """
    scenario = generate_synthetic_scenario(base_config, np.random.default_rng(5), scenario_id="s0")
    spawn_multiplier = scenario.perturbation_multipliers["order_spawn_rate"]
    expected = tuple(round(v * spawn_multiplier, 4) for v in base_config.demand_profile)
    assert scenario.env_config.demand_profile == expected


def test_order_expiry_steps_stays_a_valid_positive_integer(base_config):
    batch = generate_synthetic_scenario_batch(base_config, n_scenarios=50, seed=11)
    for scenario in batch:
        assert isinstance(scenario.env_config.order_expiry_steps, int)
        assert scenario.env_config.order_expiry_steps >= 1


def test_to_dict_is_json_serializable(base_config):
    import json

    scenario = generate_synthetic_scenario(base_config, np.random.default_rng(3), scenario_id="s0")
    json.dumps(scenario.to_dict())  # raises if anything isn't serializable


def test_scenario_is_a_genuine_env_config(base_config):
    """The generated config must actually be usable to build a real FleetDispatchEnv."""
    from env.fleet_env import FleetDispatchEnv

    scenario = generate_synthetic_scenario(base_config, np.random.default_rng(2), scenario_id="s0")
    env = FleetDispatchEnv(scenario.env_config)
    obs, infos = env.reset(seed=1)
    assert len(obs) == base_config.n_agents
