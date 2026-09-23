"""
Simulator/training-distribution overfitting diagnostic (Phase 2,
addition 8): flags a policy that scores well on the exact scenario it
was trained/tuned against but degrades disproportionately once
evaluated on scenarios it has never seen, versus a policy that is
genuinely robust to the kind of variation the environment can produce.

Why completion_ratio (not raw completions) is the comparison metric:
raw order-completion counts are not comparable across scenarios with
different demand volume or payout economics -- a harder synthetic
scenario should legitimately complete fewer orders even for a perfect
policy. `evaluation.report.evaluate_policy` already normalizes for
exactly this by dividing by a *centralized-optimal baseline computed
for that same scenario's config* (evaluation/centralized_baseline.py),
so `completion_ratio` is a difficulty-controlled score: "how close to
optimal was this policy, for however hard this particular scenario
happened to be." Comparing that normalized score across the base
scenario and a batch of held-out synthetic scenarios (see
training/synthetic_scenarios.py) isolates *generalization*, not just
"harder scenarios score lower."

This module deliberately reuses training/synthetic_scenarios.py rather
than inventing a second notion of scenario perturbation: the same
scalar economic/demand perturbations used to diversify training are
used here, unperturbed in structure (grid/agents/action-space never
change -- see that module's own docstring), as the held-out
distribution a genuinely non-overfit policy should transfer to.
"""
from dataclasses import dataclass, field

import numpy as np

from env.config import EnvConfig
from evaluation.episode_runner import ActionFn
from evaluation.report import evaluate_policy
from training.synthetic_scenarios import generate_synthetic_scenario_batch

# Documented threshold: if the base scenario's completion_ratio exceeds
# the mean completion_ratio across held-out synthetic scenarios by more
# than this many absolute points, the policy is flagged as likely
# overfit to the exact base distribution rather than genuinely robust.
# Chosen as a round number comfortably above the run-to-run noise
# observed for a single policy re-evaluated on the same scenario
# (typically a few points of completion_ratio at n_episodes_per_scenario=5);
# tightening it trades false negatives for false positives on a
# noisier evaluation budget.
DEFAULT_GAP_THRESHOLD = 0.15


@dataclass
class GeneralizationGapReport:
    """Base-vs-synthetic generalization comparison for one policy."""

    base_completion_ratio: float
    synthetic_completion_ratios: list[float]
    scenario_ids: list[str]
    mean_synthetic_completion_ratio: float
    std_synthetic_completion_ratio: float
    generalization_gap: float
    gap_threshold: float
    flagged: bool

    def to_dict(self) -> dict:
        return {
            "base_completion_ratio": self.base_completion_ratio,
            "synthetic_completion_ratios": self.synthetic_completion_ratios,
            "scenario_ids": self.scenario_ids,
            "mean_synthetic_completion_ratio": self.mean_synthetic_completion_ratio,
            "std_synthetic_completion_ratio": self.std_synthetic_completion_ratio,
            "generalization_gap": self.generalization_gap,
            "gap_threshold": self.gap_threshold,
            "flagged": self.flagged,
        }

    def summary(self) -> str:
        status = "FLAGGED -- likely overfit to the base scenario" if self.flagged else "OK -- generalizes within tolerance"
        return (
            f"Generalization check: {status}\n"
            f"  base completion_ratio:               {self.base_completion_ratio:.4f}\n"
            f"  synthetic mean completion_ratio:      {self.mean_synthetic_completion_ratio:.4f} "
            f"(std {self.std_synthetic_completion_ratio:.4f}, n={len(self.synthetic_completion_ratios)})\n"
            f"  generalization gap (base - synthetic): {self.generalization_gap:.4f} "
            f"(threshold {self.gap_threshold:.4f})"
        )


def classify_generalization_gap(
    base_ratio: float, synthetic_ratios: list[float], threshold: float = DEFAULT_GAP_THRESHOLD
) -> tuple[float, float, float, bool]:
    """
    Pure aggregation logic, independently testable without running any
    episodes: given a base completion_ratio and a batch of synthetic
    completion_ratios, return
    (mean_synthetic, std_synthetic, generalization_gap, flagged).
    """
    synthetic_array = np.asarray(synthetic_ratios, dtype=np.float64)
    mean_synth = float(np.mean(synthetic_array)) if synthetic_array.size else 0.0
    std_synth = float(np.std(synthetic_array)) if synthetic_array.size else 0.0
    gap = base_ratio - mean_synth
    flagged = gap > threshold
    return mean_synth, std_synth, gap, flagged


def run_generalization_check(
    action_fn: ActionFn,
    base_config: EnvConfig | None = None,
    n_synthetic_scenarios: int = 20,
    synthetic_seed: int = 42,
    n_episodes_per_scenario: int = 5,
    base_seed: int = 9000,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> GeneralizationGapReport:
    """
    Evaluate `action_fn` on `base_config` and on `n_synthetic_scenarios`
    held-out synthetic variants of it, and classify the resulting
    generalization gap. Each scenario (base and synthetic) gets its
    own non-overlapping seed range so no episode is shared between
    them.
    """
    base_config = base_config or EnvConfig()

    base_report = evaluate_policy(base_config, action_fn, n_episodes=n_episodes_per_scenario, base_seed=base_seed)
    base_ratio = base_report.efficiency.completion_ratio

    scenarios = generate_synthetic_scenario_batch(base_config, n_scenarios=n_synthetic_scenarios, seed=synthetic_seed)

    synthetic_ratios = []
    seed_block = n_episodes_per_scenario + 1  # +1 keeps a gap so adjacent blocks never touch
    for i, scenario in enumerate(scenarios):
        scenario_seed = base_seed + (i + 1) * seed_block * 10
        report = evaluate_policy(
            scenario.env_config, action_fn, n_episodes=n_episodes_per_scenario, base_seed=scenario_seed
        )
        synthetic_ratios.append(report.efficiency.completion_ratio)

    mean_synth, std_synth, gap, flagged = classify_generalization_gap(base_ratio, synthetic_ratios, gap_threshold)

    return GeneralizationGapReport(
        base_completion_ratio=base_ratio,
        synthetic_completion_ratios=synthetic_ratios,
        scenario_ids=[s.scenario_id for s in scenarios],
        mean_synthetic_completion_ratio=mean_synth,
        std_synthetic_completion_ratio=std_synth,
        generalization_gap=gap,
        gap_threshold=gap_threshold,
        flagged=flagged,
    )
