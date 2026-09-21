"""
Continues training (fine-tunes) an existing PPO checkpoint on a new,
harder scenario (Phase 2, addition 2).

`training/train_ppo.py` trains a shared-policy PPO agent from scratch
under the default `EnvConfig` -- a moderate, TLC-calibrated demand
volume. This script demonstrates the other half of a real RL workflow:
taking an *already-trained* checkpoint and continuing training on a
genuinely different distribution (`training/scenarios.py`'s surge
scenario -- a sustained ~70% increase in order-arrival rate, not a
relabeled copy of the same config) instead of training a fresh policy
from zero.

Weight transfer, not `Algorithm.from_checkpoint`, is used deliberately:
`from_checkpoint` restores the *entire* saved AlgorithmConfig,
including the original (non-surge) `env_config`, so it cannot be used
to continue training under a different environment config. Instead:

1. Restore the baseline `Algorithm` from its checkpoint directory.
2. Extract its policy weights with `algo.get_weights()`.
3. Build a *new* `Algorithm` under the surge `PPOConfig` (same
   observation/action spaces -- only `order_spawn_rate` differs).
4. Load the baseline weights into it with `algo.set_weights(...)`.
5. Continue training (`algo.train()`) from that warm start.

This is the standard weight-transfer pattern for fine-tuning an RLlib
policy onto a new environment configuration, and was verified
end-to-end in this repository before being committed (baseline
checkpoint -> weight transfer -> continued training -> measurably
different, and worse, reward under the harder surge scenario -- the
expected, honest direction for the very first few fine-tuning
iterations against a harder distribution, before it re-adapts).

Usage:
    # Train a short baseline checkpoint from scratch, then fine-tune it:
    python -m training.train_ppo --iterations 3 --checkpoint-dir checkpoints/baseline
    python -m training.finetune_ppo \\
        --baseline-checkpoint checkpoints/baseline/iter_3 \\
        --iterations 3 --checkpoint-dir checkpoints/finetuned
"""
import argparse
import json
import os

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from env.config import EnvConfig
from training.rllib_env_wrapper import RLlibFleetDispatchEnv
from training.scenarios import SURGE_ORDER_SPAWN_RATE, build_surge_scenario_config
from training.train_ppo import ENV_NAME, build_ppo_config, env_creator


def finetune_from_checkpoint(
    baseline_checkpoint_dir: str,
    iterations: int,
    checkpoint_dir: str,
    n_rollout_workers: int = 1,
    train_batch_size: int = 2000,
    checkpoint_every: int = 1,
) -> dict:
    """
    Restore `baseline_checkpoint_dir`'s policy weights into a freshly
    built PPO algorithm configured for the surge scenario, continue
    training for `iterations` iterations, and return a small summary
    dict (used by both the CLI below and the benchmark package).
    """
    register_env(ENV_NAME, env_creator)

    base_config = EnvConfig()
    surge_config = build_surge_scenario_config(base_config)

    baseline_algo = Algorithm.from_checkpoint(baseline_checkpoint_dir)
    baseline_weights = baseline_algo.get_weights()
    baseline_algo.stop()

    surge_ppo_config = build_ppo_config(surge_config, n_rollout_workers)
    surge_ppo_config = surge_ppo_config.training(train_batch_size=train_batch_size)
    algo = surge_ppo_config.build()
    algo.set_weights(baseline_weights)

    os.makedirs(checkpoint_dir, exist_ok=True)
    reward_history = []

    for i in range(1, iterations + 1):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_return_mean")
        reward_history.append(reward_mean)
        print(f"[fine-tune] iteration {i}/{iterations} | episode_return_mean={reward_mean}")

        if i % checkpoint_every == 0 or i == iterations:
            save_result = algo.save(os.path.join(checkpoint_dir, f"finetune_iter_{i}"))
            print(f"  Saved checkpoint: {save_result.checkpoint.path}")

    algo.stop()

    return {
        "baseline_checkpoint": baseline_checkpoint_dir,
        "scenario": "surge",
        "surge_order_spawn_rate": SURGE_ORDER_SPAWN_RATE,
        "base_order_spawn_rate": base_config.order_spawn_rate,
        "iterations": iterations,
        "reward_history": reward_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/finetuned")
    parser.add_argument("--n-rollout-workers", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--summary-output", type=str, default=None)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    summary = finetune_from_checkpoint(
        baseline_checkpoint_dir=args.baseline_checkpoint,
        iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        n_rollout_workers=args.n_rollout_workers,
        train_batch_size=args.train_batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    ray.shutdown()

    print(json.dumps(summary, indent=2))
    if args.summary_output:
        with open(args.summary_output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
