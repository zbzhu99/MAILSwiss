import numpy as np
from gym.spaces import Box, Dict, Discrete, Tuple

from marlkit.core import PrefixDict
from marlkit.data_management.simple_replay_buffer import (
    AgentMetaSimpleReplayBuffer,
    AgentSimpleReplayBuffer,
)


class EnvReplayBuffer:
    def __init__(self, max_replay_buffer_size, env, use_prefix_dict: bool = False):
        self._observation_space_n = env.observation_space_n
        self._action_space_n = env.action_space_n
        self.n_agents = env.n_agents
        self.agent_ids = env.agent_ids
        if use_prefix_dict:
            dict_cls = PrefixDict
        else:
            dict_cls = dict
        self.agent_buffer_n = dict_cls(
            zip(
                self.agent_ids,
                [
                    AgentEnvReplayBuffer(
                        max_replay_buffer_size,
                        self._observation_space_n[a_id],
                        self._action_space_n[a_id],
                    )
                    for a_id in self.agent_ids
                ]
            )
        )
        self._max_replay_buffer_size = max_replay_buffer_size

    def num_steps_can_sample(self):
        return list(self.agent_buffer_n.values())[0].num_steps_can_sample()

    def random_batch(self, batch_size: int, agent_id: str, keys=None):
        return self.agent_buffer_n[agent_id].random_batch(batch_size, keys)

    def terminate_episode(self, agent_id: str):
        self.agent_buffer_n[agent_id].terminate_episode()

    def sample_trajs(
        self, num_trajs: int, agent_id: str, keys=None, samples_per_traj=None
    ):
        return self.agent_buffer_n[agent_id].sample_trajs(
            num_trajs, keys, samples_per_traj
        )

    def clear(self, agent_id: str):
        self.agent_buffer_n[agent_id].clear()

    def add_path(self, path_n, absorbing: bool = False, env=None):
        for a_id in self.agent_ids:
            self.agent_buffer_n[a_id].add_path(
                path_n[a_id], absorbing=absorbing, env=env
            )

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        agent_id,
        **kwargs,
    ):
        self.agent_buffer_n[agent_id].add_sample(
            observation,
            action,
            reward,
            terminal,
            next_observation,
            **kwargs,
        )


class AgentEnvReplayBuffer(AgentSimpleReplayBuffer):
    def __init__(
        self, max_replay_buffer_size, observation_space, action_space, random_seed=1995
    ):
        """
        :param max_replay_buffer_size:
        :param observation_space:
        :param action_space:
        """
        self._ob_space = observation_space
        self._action_space = action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        super(AgentEnvReplayBuffer, self).add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )


class AgentMetaEnvReplayBuffer(AgentMetaSimpleReplayBuffer):
    def __init__(
        self,
        max_rb_size_per_task,
        env,
        extra_obs_dim=0,
        policy_uses_pixels=False,
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        assert extra_obs_dim == 0, "I removed the extra_obs_dim thing"
        # self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_rb_size_per_task=max_rb_size_per_task,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            policy_uses_pixels=policy_uses_pixels,
            policy_uses_task_params=policy_uses_task_params,
            concat_task_params_to_policy_obs=concat_task_params_to_policy_obs,
        )

    def add_path(self, path, task_identifier):
        if isinstance(self._action_space, Discrete):
            action_array = np.eye(self._action_space.n)[path["actions"][:, 0]]
            path["actions"] = action_array
        super(AgentMetaEnvReplayBuffer, self).add_path(path, task_identifier)


def get_dim(space):
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        # return space.n
        return 1
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
