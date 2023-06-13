import abc

from marlkit.env_creators import GymEnv, MpeEnv, MujocoEnv
from marlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from marlkit.envs.wrappers import ProxyEnv

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]


def get_env(env_specs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    env_creator = env_specs["env_creator"]

    if env_creator == "mujoco":
        env_class = MujocoEnv
    elif env_creator == "mpe":
        env_class = MpeEnv
    elif env_creator == "gym":
        env_class = GymEnv
    else:
        raise NotImplementedError

    env = env_class(**env_specs)

    return env


def get_envs(
    env_specs,
    env_wrapper=None,
    env_num=1,
    **kwargs,
):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """

    if env_wrapper is None:
        env_wrapper = ProxyEnv

    env_creator = env_specs["env_creator"]

    if env_creator == "mujoco":
        env_class = MujocoEnv
    elif env_creator == "mpe":
        env_class = MpeEnv
    elif env_creator == "gym":
        env_class = GymEnv
    else:
        raise NotImplementedError

    if env_num == 1:
        envs = env_wrapper(env_class(**env_specs))

        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.")
        envs = DummyVectorEnv(
            [lambda: envs],
            **kwargs,
        )

    else:
        envs = SubprocVectorEnv(
            [lambda: env_wrapper(env_class(**env_specs)) for _ in range(env_num)],
            **kwargs,
        )

    return envs


class EnvFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __get__(self, task_params):
        """
        Implements returning and environment corresponding to given task params
        """
        pass

    @abc.abstractmethod
    def get_task_identifier(self, task_params):
        """
        Returns a hashable description of task params so it can be used
        as dictionary keys etc.
        """
        pass

    def task_params_to_obs_task_params(self, task_params):
        """
        Sometimes this may be needed. For example if we are training a
        multitask RL algorithm and want to give it the task params as
        part of the state.
        """
        raise NotImplementedError()
