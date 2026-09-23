"""
CLI entry point for the simulator-overfitting / generalization-gap
diagnostic (Phase 2, addition 8 -- see evaluation/generalization_gap.py
for the methodology).

Usage:
    # Smoke test with the random policy (no trained checkpoint needed):
    python -m evaluation.run_generalization_check --n-synthetic-scenarios 10

    # Against a trained checkpoint:
    python -m evaluation.run_generalization_check --checkpoint checkpoints/iter_50 \\
        --n-synthetic-scenarios 20 --output benchmark_results/generalization_check.json
"""
import argparse
import json

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.episode_runner import ActionFn, random_action_fn
from evaluation.generalization_gap import DEFAULT_GAP_THRESHOLD, run_generalization_check


def _load_checkpoint_action_fn(checkpoint_path: str) -> ActionFn:
    """Same weight-restore pattern as evaluation/run_quality_gate.py -- imported lazily so Ray/RLlib
    aren't required just to smoke-test with the random policy."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-synthetic-scenarios", type=int, default=20)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--n-episodes-per-scenario", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    action_fn = _load_checkpoint_action_fn(args.checkpoint) if args.checkpoint else random_action_fn

    report = run_generalization_check(
        action_fn,
        base_config=EnvConfig(),
        n_synthetic_scenarios=args.n_synthetic_scenarios,
        synthetic_seed=args.synthetic_seed,
        n_episodes_per_scenario=args.n_episodes_per_scenario,
        base_seed=args.base_seed,
        gap_threshold=args.gap_threshold,
    )
    print(report.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved report to {args.output}")

    if report.flagged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
