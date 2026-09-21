"""
CLI entry point for the automated QA / quality gate (Phase 2, addition 1).

Runs a policy (a trained RLlib checkpoint, or the random policy as a
smoke test) through a batch of episodes, computes the automated
run-health checks in `evaluation/quality_gate.py`, and prints/saves a
QualityGateReport -- the step a real training pipeline would run after
every N iterations, automatically, before anything is ever queued for
a human rater in `annotation_app/`.

Usage:
    # Smoke test with the random policy (no trained checkpoint needed):
    python -m evaluation.run_quality_gate --n-episodes 10

    # Against a trained checkpoint:
    python -m evaluation.run_quality_gate --checkpoint checkpoints/iter_50 --n-episodes 10
"""
import argparse
import json

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.centralized_baseline import run_centralized_baseline_episode
from evaluation.episode_runner import ActionFn, random_action_fn, run_episode
from evaluation.quality_gate import run_quality_gate


def _load_checkpoint_action_fn(checkpoint_path: str) -> ActionFn:
    """
    Build an ActionFn from a saved RLlib checkpoint (see
    training/train_ppo.py for how checkpoints are produced). Imported
    lazily so `--n-episodes` smoke tests with the random policy don't
    require Ray/RLlib to be importable.
    """
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
            logits = out["action_dist_inputs"]
            actions[aid] = int(torch.argmax(logits, dim=-1).item())
        return actions

    return action_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved RLlib checkpoint.")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--run-id", type=str, default="manual-run")
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the report as JSON.")
    args = parser.parse_args()

    config = EnvConfig()
    action_fn = _load_checkpoint_action_fn(args.checkpoint) if args.checkpoint else random_action_fn

    traces = []
    baseline_completions = 0
    for i in range(args.n_episodes):
        seed = args.base_seed + i
        env = FleetDispatchEnv(config)
        traces.append(run_episode(env, action_fn, seed=seed))
        baseline_result = run_centralized_baseline_episode(config, seed=seed)
        baseline_completions += baseline_result.total_completed

    report = run_quality_gate(run_id=args.run_id, traces=traces, baseline_completions=baseline_completions)
    print(report.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved report to {args.output}")

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
