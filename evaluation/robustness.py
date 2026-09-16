"""
Robustness evaluation under demand spikes, per the design document's
"Robustness: performance under demand spikes (rush hour, weather)
without retraining."

Implemented by running the same policy (no retraining, no parameter
changes) against a modified EnvConfig with an artificially amplified
demand profile, and comparing completion counts against a matched
normal-demand run.
"""
import dataclasses

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.episode_runner import ActionFn, run_episode
from evaluation.metrics import RobustnessMetrics


def build_spike_config(base_config: EnvConfig, spike_multiplier: float = 2.5) -> EnvConfig:
    """
    Return a new EnvConfig with every hourly demand multiplier scaled
    up by `spike_multiplier` -- simulating a sustained city-wide demand
    spike (e.g. severe weather, a major event) rather than just a
    single busy hour, so a policy trained on normal demand faces
    meaningfully higher pressure for the entire episode.
    """
    spiked_profile = tuple(round(m * spike_multiplier, 3) for m in base_config.demand_profile)
    return dataclasses.replace(base_config, demand_profile=spiked_profile)


def run_robustness_test(
    base_config: EnvConfig, action_fn: ActionFn, seed: int, spike_multiplier: float = 2.5
) -> RobustnessMetrics:
    """Run the same policy under normal and spiked demand (same seed for a fair comparison) and compare."""
    normal_env = FleetDispatchEnv(base_config)
    normal_trace = run_episode(normal_env, action_fn, seed=seed)

    spike_config = build_spike_config(base_config, spike_multiplier)
    spike_env = FleetDispatchEnv(spike_config)
    spike_trace = run_episode(spike_env, action_fn, seed=seed)

    return RobustnessMetrics.compute(
        normal_completions=normal_trace.total_completed,
        spike_completions=spike_trace.total_completed,
    )
