"""
Trains a shared-parameter PPO policy across all driver-agents in
FleetDispatchEnv using RLlib.

Parameter sharing (one policy, applied identically to every agent) is
the standard approach for homogeneous multi-agent settings like this
one -- every driver-agent has the same action/observation space and
the same reward structure, so sharing weights is both more
sample-efficient than training 8 separate policies and avoids each
agent overfitting to the specific identities of its 7 fleet-mates.

Usage:
    python -m training.train_ppo --iterations 50 --checkpoint-dir checkpoints/
"""
import argparse
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from env.config import EnvConfig
from training.rllib_env_wrapper import RLlibFleetDispatchEnv

ENV_NAME = "fleet_dispatch_v0"


def env_creator(config: dict) -> MultiAgentEnv:
    return RLlibFleetDispatchEnv(config)


def build_ppo_config(env_config: EnvConfig, n_rollout_workers: int = 2) -> PPOConfig:
    """
    Build the PPOConfig for training, with a single shared policy
    ("shared_policy") mapped to every agent_id.
    """
    dummy_env = RLlibFleetDispatchEnv({"env_config": env_config})
    obs_space = dummy_env.observation_spaces["driver_0"]
    act_space = dummy_env.action_spaces["driver_0"]

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config={"env_config": env_config})
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=n_rollout_workers, rollout_fragment_length="auto")
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            lr=3e-4,
            gamma=0.97,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
        )
        .framework("torch")
        .resources(num_gpus=0)
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--n-rollout-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    register_env(ENV_NAME, env_creator)
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    env_config = EnvConfig()
    ppo_config = build_ppo_config(env_config, args.n_rollout_workers)
    algo = ppo_config.build()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for i in range(1, args.iterations + 1):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_return_mean")
        print(f"Iteration {i}/{args.iterations} | episode_return_mean={reward_mean}")

        if i % args.checkpoint_every == 0 or i == args.iterations:
            save_result = algo.save(os.path.join(args.checkpoint_dir, f"iter_{i}"))
            checkpoint_path = save_result.checkpoint.path
            print(f"  Saved checkpoint: {checkpoint_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
