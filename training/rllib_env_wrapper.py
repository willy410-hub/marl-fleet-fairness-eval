"""
Wraps FleetDispatchEnv (a PettingZoo ParallelEnv) as an RLlib
MultiAgentEnv, so it can be registered and trained with RLlib's PPO
implementation.
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv


class RLlibFleetDispatchEnv(MultiAgentEnv):
    """Thin adapter: PettingZoo's ParallelEnv API -> RLlib's MultiAgentEnv API."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        env_config = config.get("env_config", EnvConfig())
        self._env = FleetDispatchEnv(env_config)

        self.agents = self.possible_agents = list(self._env.possible_agents)
        self.observation_spaces = {
            aid: self._env.observation_space(aid) for aid in self.possible_agents
        }
        self.action_spaces = {aid: self._env.action_space(aid) for aid in self.possible_agents}

    def reset(self, *, seed=None, options=None):
        obs, infos = self._env.reset(seed=seed, options=options)
        return obs, infos

    def step(self, action_dict):
        obs, rewards, terminations, truncations, infos = self._env.step(action_dict)

        terminations["__all__"] = len(self._env.agents) == 0 and all(terminations.values()) if terminations else False
        truncations["__all__"] = len(self._env.agents) == 0

        return obs, rewards, terminations, truncations, infos

    @property
    def unwrapped_env(self) -> FleetDispatchEnv:
        """Access the underlying FleetDispatchEnv (used by evaluation/ for introspection)."""
        return self._env
