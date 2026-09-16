"""
Top-level evaluation report generator: runs a policy through every
metric defined in Section 4 of the design document and produces one
consolidated, JSON-serializable report.
"""
import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.centralized_baseline import run_centralized_baseline_episode
from evaluation.episode_runner import ActionFn, run_episode
from evaluation.metrics import DecisionQualityMetrics, EfficiencyMetrics, FairnessMetrics
from evaluation.robustness import run_robustness_test


@dataclass
class EvaluationReport:
    """The full, consolidated evaluation output for one policy, across n_episodes."""

    n_episodes: int
    efficiency: EfficiencyMetrics
    fairness: FairnessMetrics
    decision_quality: DecisionQualityMetrics
    robustness_2x: object
    robustness_2_5x: object
    per_episode_completions: list = field(default_factory=list)
    per_episode_rewards: list = field(default_factory=list)

    def to_dict(self) -> dict:
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        return {
            "n_episodes": self.n_episodes,
            "efficiency": convert(self.efficiency),
            "fairness": convert(self.fairness),
            "decision_quality": convert(self.decision_quality),
            "robustness_2x_spike": convert(self.robustness_2x),
            "robustness_2_5x_spike": convert(self.robustness_2_5x),
            "per_episode_completions": self.per_episode_completions,
            "per_episode_rewards": self.per_episode_rewards,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_policy(
    config: EnvConfig, action_fn: ActionFn, n_episodes: int = 10, base_seed: int = 1000
) -> EvaluationReport:
    """
    Run `action_fn` through `n_episodes` full episodes and every
    metric in Section 4 of the design document, returning one
    consolidated report.
    """
    all_completions = []
    all_rewards = []
    all_zone_completions = np.zeros(len(FleetDispatchEnv(config).zones), dtype=np.int64)
    all_zone_offers = np.zeros_like(all_zone_completions)
    all_zone_rejections = np.zeros_like(all_zone_completions)
    all_accepted_flags = []
    all_offer_payouts = []
    all_baseline_completions = []

    for i in range(n_episodes):
        seed = base_seed + i
        env = FleetDispatchEnv(config)
        trace = run_episode(env, action_fn, seed=seed)

        all_completions.append(trace.total_completed)
        all_rewards.append(trace.total_reward)
        all_zone_completions += trace.zone_completion_counts
        all_zone_offers += trace.zone_offer_counts
        all_zone_rejections += trace.zone_rejection_counts
        all_accepted_flags.extend(trace.accepted_flags)
        all_offer_payouts.extend(trace.offer_payouts)

        baseline_result = run_centralized_baseline_episode(config, seed=seed)
        all_baseline_completions.append(baseline_result.total_completed)

    efficiency = EfficiencyMetrics.compute(
        policy_completions=int(np.sum(all_completions)),
        baseline_completions=int(np.sum(all_baseline_completions)),
    )
    fairness = FairnessMetrics.compute(all_zone_completions, all_zone_rejections, all_zone_offers)
    decision_quality = DecisionQualityMetrics.compute(
        np.array(all_accepted_flags), np.array(all_offer_payouts)
    )
    robustness_2x = run_robustness_test(config, action_fn, seed=base_seed, spike_multiplier=2.0)
    robustness_2_5x = run_robustness_test(config, action_fn, seed=base_seed, spike_multiplier=2.5)

    return EvaluationReport(
        n_episodes=n_episodes,
        efficiency=efficiency,
        fairness=fairness,
        decision_quality=decision_quality,
        robustness_2x=robustness_2x,
        robustness_2_5x=robustness_2_5x,
        per_episode_completions=all_completions,
        per_episode_rewards=all_rewards,
    )
