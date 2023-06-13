"""
Torch argmax policy
"""
import numpy as np

from marlkit.policies.base import ExplorationPolicy
from marlkit.torch.core import PyTorchModule


class ArgmaxDiscretePolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, qf, action_space, epsilon=0.1):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf
        self.action_num = action_space.n
        self.prob_random_action = epsilon

    def get_actions(self, obs_np, deterministic=False):
        if deterministic or np.random.random() >= self.prob_random_action:
            q_values = self.qf.eval_np(obs_np)
            return np.expand_dims(q_values.argmax(), -1)
        else:
            return np.random.randint(self.action_num, size=obs_np.shape[0])

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action[0], {}
