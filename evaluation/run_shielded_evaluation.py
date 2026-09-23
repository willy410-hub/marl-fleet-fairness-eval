"""
CLI entry point demonstrating the rule-based safety shield
(Phase 2, addition 6, env/safety_shield.py): runs a policy (a trained
RLlib checkpoint, or the random policy as a smoke test) both
unshielded and with the hard-constraint shortcut ban enabled, and
reports how many actions the shield had to override plus how the
headline efficiency/fairness metrics moved -- the concrete evidence
that the shield changes real behavior, not just a unit-tested
abstraction.

Usage:
    # Smoke test with the random policy (no trained checkpoint needed):
    python -m evaluation.run_shielded_evaluation --n-episodes 5

    # Against a trained checkpoint:
    python -m evaluation.run_shielded_evaluation --checkpoint checkpoints/iter_50 --n-episodes 5
"""
import argparse
import json

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from env.safety_shield import ShieldedActionFn
from evaluation.episode_runner import ActionFn, random_action_fn
from evaluation.report import evaluate_policy


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
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=8000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = EnvConfig()
    base_action_fn = _load_checkpoint_action_fn(args.checkpoint) if args.checkpoint else random_action_fn

    unshielded_report = evaluate_policy(config, base_action_fn, n_episodes=args.n_episodes, base_seed=args.base_seed)

    shield = ShieldedActionFn(base_action_fn, disallow_shortcuts=True)
    shielded_report = evaluate_policy(config, shield, n_episodes=args.n_episodes, base_seed=args.base_seed)

    print("=== Unshielded ===")
    print(f"  completion_ratio: {unshielded_report.efficiency.completion_ratio:.3f}")
    print(f"  zone_gini:        {unshielded_report.fairness.zone_gini:.3f}")

    print("\n=== Shielded (shortcut moves hard-disallowed) ===")
    print(f"  completion_ratio: {shielded_report.efficiency.completion_ratio:.3f}")
    print(f"  zone_gini:        {shielded_report.fairness.zone_gini:.3f}")
    print(f"  shield overrides: {shield.override_count} action(s) across {args.n_episodes} episode(s)")

    if args.output:
        payload = {
            "unshielded": unshielded_report.to_dict(),
            "shielded": shielded_report.to_dict(),
            "shield_override_count": shield.override_count,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
