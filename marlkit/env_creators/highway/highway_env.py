import gymnasium

from marlkit.env_creators.base_env import BaseEnv


class HighwayEnv(BaseEnv):
    """A wrapper for highway env to fit in multi-agent apis."""

    def __init__(self, **configs):
        super().__init__(**configs)

        # create underlying mujoco env
        env_name = configs["env_name"]
        env_kwargs = configs["env_kwargs"]
        self._env = gymnasium.make(env_name, **env_kwargs)

        self._default_agent_name = "agent_0"
        self._default_agent_ID = 0

        #self.agent_ids = [self._default_agent_name]
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = {
            self._default_agent_name: self._env.observation_space
        }
        self.action_space_n = {self._default_agent_name: self._env.action_space}

    def seed(self, seed):
        if hasattr(self._env, "seed"):
            return self._env.seed(seed)
        else:
            return self._env.reset(seed=seed)

    def reset(self):
        return self._env.reset()

    def step(self, action_n):
        action = action_n[self._default_agent_ID]
        next_obs, rew, done, trunc, info = self._env.step(action)

        self.n_agents = len(next_obs)
        
        next_obs_n, rew_n, done_n, trunc_n, info_n = {}, {}, {}, {}, {}
        for agent_id in range(self.n_agents):
            next_obs_n.update({agent_id: next_obs[agent_id]})
            rew_n.update({agent_id: rew[agent_id]})
            done_n.update({agent_id: done[agent_id]})
            trunc_n.update({agent_id: trunc[agent_id]})
            info_n.update({agent_id: info[agent_id]})
        return next_obs_n, rew_n, done_n, trunc_n, info_n

    def render(self, **kwargs):
        return self._env.render(**kwargs)

