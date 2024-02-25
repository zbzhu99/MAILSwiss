from typing import Dict
import gym
import gymnasium
import numpy as np

from marlkit.env_creators.base_env import BaseEnv


def to_gym_space(space, flatten: bool = False):
    if isinstance(space, gym.spaces.Space):
        return space
    else:
        if isinstance(space, gymnasium.spaces.Box):
            if flatten:
                high = space.high.reshape(-1)
                low = space.low.reshape(-1)
                shape = (np.prod(space.shape),)
            else:
                high = space.high
                low = space.low
                shape = space.shape
            return gym.spaces.Box(
                high=high, low=low, shape=shape, dtype=space.dtype
            )
        elif isinstance(space, gymnasium.spaces.Discrete):
            return gym.spaces.Discrete(n=space.n)
        else:
            raise NotImplementedError(f"Unsupported space type: {type(space)}")


class HighwayEnv(BaseEnv):
    """A wrapper for highway env to fit in multi-agent apis."""

    def __init__(self, **configs):
        super().__init__(**configs)

        env_name = configs["env_name"]
        env_kwargs = configs["env_kwargs"]
        self._env = gymnasium.make(env_name, disable_env_checker=True, **env_kwargs)

        self.agent_ids = ["main", "merging"]
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = {
            agent_id: to_gym_space(self._env.observation_space, flatten=True) for agent_id in self.agent_ids
        }
        self.action_space_n = {
            agent_id: to_gym_space(self._env.action_space, flatten=True) for agent_id in self.agent_ids
        }
        self.prev_vehicle_ids = None

    def seed(self, seed):
        if hasattr(self._env, "seed"):
            return self._env.unwrapped.seed(seed)
        else:
            return self._env.reset(seed=seed)

    def reset(self):
        obs_n_list, info = self._env.reset()
        self.prev_vehicle_ids = info["vehicle_ids"]
        obs_n = {}
        for obs, v_id in zip(obs_n_list, info["vehicle_ids"]):
            obs_n[v_id] = obs.reshape(-1)
        return obs_n

    def step(self, action_n: Dict[str, np.ndarray]):
        action_n_list = np.array([action_n[v_id] for v_id in self.prev_vehicle_ids])
        next_obs_n_list, rew_n_list, done_n_list, _, info = self._env.step(action_n_list)
        self.prev_vehicle_ids = info["vehicle_ids"]
        next_obs_n, rew_n, done_n, info_n = {}, {}, {}, {}
        for idx, v_id in enumerate(info["vehicle_ids"]):
            next_obs_n[v_id] = next_obs_n_list[idx].reshape(-1)
            rew_n[v_id] = rew_n_list[idx]
            done_n[v_id] = done_n_list[idx]
            info_n[v_id] = {}
        return next_obs_n, rew_n, done_n, info_n

    def render(self, **kwargs):
        return self._env.render(**kwargs)
