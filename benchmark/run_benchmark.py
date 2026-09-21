"""
Benchmark suite runner (Phase 2, addition 4).

Scores a policy (a trained RLlib checkpoint, or the random baseline as
a reference point) against every fixed task in `benchmark/tasks.py`,
reusing the existing evaluation machinery
(`evaluation.report.evaluate_policy`) and the automated quality gate
(`evaluation.quality_gate.run_quality_gate`) rather than duplicating
either -- this package is a thin, reusable wrapper around code that
already exists and is already tested, which is the whole point of
"packaging it as a reusable benchmark" instead of writing a second,
parallel evaluation path.

Usage:
    # Score the random baseline against every task (no checkpoint needed):
    python -m benchmark.run_benchmark --output benchmark_results/random_baseline.json

    # Score a trained checkpoint:
    python -m benchmark.run_benchmark --checkpoint checkpoints/iter_50 \\
        --output benchmark_results/my_policy.json
"""
import argparse
import json
from dataclasses import asdict, dataclass

from env.fleet_env import FleetDispatchEnv
from evaluation.centralized_baseline import run_centralized_baseline_episode
from evaluation.episode_runner import ActionFn, random_action_fn, run_episode
from evaluation.quality_gate import QualityGateReport, run_quality_gate
from evaluation.report import EvaluationReport, evaluate_policy
from benchmark.tasks import TASKS, BenchmarkTask


@dataclass
class TaskScore:
    task_id: str
    evaluation: EvaluationReport
    quality_gate: QualityGateReport


def _load_checkpoint_action_fn(checkpoint_path: str) -> ActionFn:
    """Same weight-restore pattern as evaluation/run_quality_gate.py -- imported lazily so Ray/RLlib
    aren't required just to score the random baseline."""
    from ray.rllib.algorithms.algorithm import Algorithm

    algo = Algorithm.from_checkpoint(checkpoint_path)
    module = algo.get_module("shared_policy")

    def action_fn(env: FleetDispatchEnv, obs: dict) -> dict[str, int]:
        import numpy as np
        import torch

        actions = {}
        for aid, agent_obs in obs.items():
            obs_tensor = torch.from_numpy(np.asarray(agent_obs, dtype=np.float32)).unsqueeze(0)
            out = module.forward_inference({"obs": obs_tensor})
            actions[aid] = int(torch.argmax(out["action_dist_inputs"], dim=-1).item())
        return actions

    return action_fn


def run_task(task: BenchmarkTask, action_fn: ActionFn) -> TaskScore:
    """Run one benchmark task: full evaluation report + automated quality gate, both on the same episodes."""
    report = evaluate_policy(task.env_config, action_fn, n_episodes=task.n_episodes, base_seed=task.base_seed)

    traces = []
    baseline_completions = 0
    for i in range(task.n_episodes):
        seed = task.base_seed + i
        env = FleetDispatchEnv(task.env_config)
        traces.append(run_episode(env, action_fn, seed=seed))
        baseline_completions += run_centralized_baseline_episode(task.env_config, seed=seed).total_completed

    gate_report = run_quality_gate(run_id=task.task_id, traces=traces, baseline_completions=baseline_completions)

    return TaskScore(task_id=task.task_id, evaluation=report, quality_gate=gate_report)


def run_full_benchmark(action_fn: ActionFn, task_ids: list[str] | None = None) -> dict[str, TaskScore]:
    """Run every task (or a subset named by `task_ids`) and return {task_id: TaskScore}."""
    ids = task_ids or list(TASKS)
    return {task_id: run_task(TASKS[task_id], action_fn) for task_id in ids}


def _score_to_dict(score: TaskScore) -> dict:
    return {
        "task_id": score.task_id,
        "evaluation": score.evaluation.to_dict(),
        "quality_gate": score.quality_gate.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Subset of task IDs (default: all).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    action_fn = _load_checkpoint_action_fn(args.checkpoint) if args.checkpoint else random_action_fn
    scores = run_full_benchmark(action_fn, task_ids=args.tasks)

    for task_id, score in scores.items():
        print(f"\n=== {task_id} ===")
        print(f"  completion_ratio (vs. centralized-optimal baseline): {score.evaluation.efficiency.completion_ratio:.3f}")
        print(f"  zone_gini: {score.evaluation.fairness.zone_gini:.3f}")
        print(f"  decision_quality (value-sensitivity correlation): {score.evaluation.decision_quality.value_sensitivity_correlation:.3f}")
        print(f"  quality_gate: {'PASSED' if score.quality_gate.passed else 'FLAGGED'}")

    if args.output:
        payload = {task_id: _score_to_dict(score) for task_id, score in scores.items()}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
