"""
Trains a PPO policy across an automatically generated curriculum of
synthetic scenarios (Phase 2, addition 5).

`training/finetune_ppo.py` demonstrates weight-transfer fine-tuning
onto ONE hand-authored scenario (the surge scenario). This script
generalizes that same weight-transfer mechanism across an entire
*generated* batch of synthetic scenarios
(`training/synthetic_scenarios.py`): train a few iterations on
scenario 0, carry the learned weights into scenario 1, and so on. The
practical effect is that a training run sees many more distinct
demand/economics variants per unit of human effort than a human
hand-authoring each scenario file could produce -- the concrete,
working version of "build automated data-generation systems ... to
accelerate training cycles without compromising quality": training
continuity (via weight transfer, not from-scratch resets) is
preserved across every synthetic scenario switch, so added diversity
doesn't cost re-convergence time at each step.

Usage:
    python -m training.train_ppo_curriculum \\
        --n-scenarios 5 --iterations-per-scenario 2 --seed 42 \\
        --checkpoint-dir checkpoints/curriculum --summary-output benchmark_results/curriculum_summary.json
"""
import argparse
import json
import os

import ray
from ray.tune.registry import register_env

from env.config import EnvConfig
from training.rllib_env_wrapper import RLlibFleetDispatchEnv
from training.synthetic_scenarios import generate_synthetic_scenario_batch
from training.train_ppo import ENV_NAME, build_ppo_config, env_creator


def train_curriculum(
    n_scenarios: int,
    seed: int,
    iterations_per_scenario: int,
    checkpoint_dir: str,
    n_rollout_workers: int = 1,
    train_batch_size: int = 2000,
) -> dict:
    """
    Train sequentially across `n_scenarios` synthetic scenarios,
    transferring policy weights forward at every switch (the same
    get_weights/set_weights pattern used in
    training/finetune_ppo.py::finetune_from_checkpoint, applied here
    in a loop instead of a single hand-picked transfer). Returns a
    summary dict with one reward-history entry per scenario.
    """
    register_env(ENV_NAME, env_creator)

    base_config = EnvConfig()
    scenarios = generate_synthetic_scenario_batch(base_config, n_scenarios=n_scenarios, seed=seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    scenario_summaries = []
    carried_weights = None

    for scenario_idx, scenario in enumerate(scenarios):
        ppo_config = build_ppo_config(scenario.env_config, n_rollout_workers)
        ppo_config = ppo_config.training(train_batch_size=train_batch_size)
        algo = ppo_config.build()

        if carried_weights is not None:
            algo.set_weights(carried_weights)

        reward_history = []
        for i in range(1, iterations_per_scenario + 1):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get("episode_return_mean")
            reward_history.append(reward_mean)
            print(
                f"[curriculum] scenario {scenario_idx + 1}/{n_scenarios} "
                f"({scenario.scenario_id}) | iteration {i}/{iterations_per_scenario} | "
                f"episode_return_mean={reward_mean}"
            )

        save_result = algo.save(os.path.join(checkpoint_dir, f"scenario_{scenario_idx:03d}"))
        print(f"  Saved checkpoint: {save_result.checkpoint.path}")

        carried_weights = algo.get_weights()
        algo.stop()

        scenario_summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "perturbation_multipliers": scenario.perturbation_multipliers,
                "reward_history": reward_history,
            }
        )

    return {
        "n_scenarios": n_scenarios,
        "seed": seed,
        "iterations_per_scenario": iterations_per_scenario,
        "scenarios": scenario_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations-per-scenario", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/curriculum")
    parser.add_argument("--n-rollout-workers", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2000)
    parser.add_argument("--summary-output", type=str, default=None)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    summary = train_curriculum(
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        iterations_per_scenario=args.iterations_per_scenario,
        checkpoint_dir=args.checkpoint_dir,
        n_rollout_workers=args.n_rollout_workers,
        train_batch_size=args.train_batch_size,
    )
    ray.shutdown()

    print(json.dumps(summary, indent=2))
    if args.summary_output:
        with open(args.summary_output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
