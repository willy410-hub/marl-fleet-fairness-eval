"""
CLI entry point: run every fixed benchmark task (benchmark/tasks.py)
across several independent seed replicates and report a mean +/- 95%
confidence interval per headline metric, plus a crisp pass/fail
verdict against the quality gate's own completion-ratio threshold
(Phase 2, addition 7 -- see benchmark/statistics.py for the
methodology this addresses).

This is additive, not a replacement for benchmark/run_benchmark.py:
that script remains the fast, single-seed path for everyday iteration;
this one is the more expensive, statistically-defensible path for a
result you intend to report or compare against someone else's.

Usage:
    # Replicate the random baseline 5x against every task:
    python -m benchmark.run_benchmark_replicated --n-replicates 5 \\
        --output benchmark_results/random_baseline_replicated.json

    # Replicate a trained checkpoint's score:
    python -m benchmark.run_benchmark_replicated --checkpoint checkpoints/iter_50 \\
        --n-replicates 5 --output benchmark_results/my_policy_replicated.json
"""
import argparse
import dataclasses
import json
from dataclasses import dataclass

from benchmark.run_benchmark import TASKS, BenchmarkTask, _load_checkpoint_action_fn, run_task
from benchmark.statistics import ConfidenceInterval, confidence_interval, crisp_pass
from evaluation.episode_runner import ActionFn, random_action_fn
from evaluation.quality_gate import MIN_COMPLETION_RATIO_VS_RANDOM

# Large enough that each replicate's episodes (base_seed .. base_seed +
# n_episodes - 1) can never overlap another replicate's, for any task
# in benchmark/tasks.py (the largest n_episodes currently defined is 10).
SEED_STRIDE = 100_000


@dataclass
class ReplicatedTaskScore:
    task_id: str
    n_replicates: int
    completion_ratio_ci: ConfidenceInterval
    zone_gini_ci: ConfidenceInterval
    quality_gate_pass_rate: float
    crisply_passes_completion_threshold: bool

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_replicates": self.n_replicates,
            "completion_ratio": self.completion_ratio_ci.to_dict(),
            "zone_gini": self.zone_gini_ci.to_dict(),
            "quality_gate_pass_rate": self.quality_gate_pass_rate,
            "crisply_passes_completion_threshold": self.crisply_passes_completion_threshold,
        }

    def summary(self) -> str:
        verdict = "CRISP PASS" if self.crisply_passes_completion_threshold else "NOT a crisp pass"
        return (
            f"[{self.task_id}] completion_ratio = {self.completion_ratio_ci} | "
            f"zone_gini = {self.zone_gini_ci} | "
            f"quality_gate_pass_rate = {self.quality_gate_pass_rate:.2f} | {verdict}"
        )


def run_task_replicated(
    task: BenchmarkTask,
    action_fn: ActionFn,
    n_replicates: int,
    completion_ratio_threshold: float = MIN_COMPLETION_RATIO_VS_RANDOM,
    seed_stride: int = SEED_STRIDE,
) -> ReplicatedTaskScore:
    """
    Run `task` `n_replicates` times, each under a non-overlapping
    seed range (`task.base_seed + r * seed_stride`), and aggregate the
    per-replicate completion_ratio and zone_gini into confidence
    intervals via benchmark/statistics.py.
    """
    completion_ratios = []
    zone_ginis = []
    gate_passes = []

    for r in range(n_replicates):
        shifted_task = dataclasses.replace(task, base_seed=task.base_seed + r * seed_stride)
        score = run_task(shifted_task, action_fn)
        completion_ratios.append(score.evaluation.efficiency.completion_ratio)
        zone_ginis.append(score.evaluation.fairness.zone_gini)
        gate_passes.append(score.quality_gate.passed)

    completion_ci = confidence_interval(completion_ratios)
    zone_gini_ci = confidence_interval(zone_ginis)

    return ReplicatedTaskScore(
        task_id=task.task_id,
        n_replicates=n_replicates,
        completion_ratio_ci=completion_ci,
        zone_gini_ci=zone_gini_ci,
        quality_gate_pass_rate=sum(gate_passes) / len(gate_passes) if gate_passes else 0.0,
        crisply_passes_completion_threshold=crisp_pass(completion_ci, completion_ratio_threshold),
    )


def run_full_benchmark_replicated(
    action_fn: ActionFn, n_replicates: int, task_ids: list[str] | None = None
) -> dict[str, ReplicatedTaskScore]:
    ids = task_ids or list(TASKS)
    return {task_id: run_task_replicated(TASKS[task_id], action_fn, n_replicates) for task_id in ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Subset of task IDs (default: all).")
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    action_fn = _load_checkpoint_action_fn(args.checkpoint) if args.checkpoint else random_action_fn
    scores = run_full_benchmark_replicated(action_fn, n_replicates=args.n_replicates, task_ids=args.tasks)

    for score in scores.values():
        print(score.summary())

    if args.output:
        payload = {task_id: score.to_dict() for task_id, score in scores.items()}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved replicated results to {args.output}")


if __name__ == "__main__":
    main()
