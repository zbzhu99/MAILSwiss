from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import marlkit.torch.utils.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.core.trainer import Trainer


class DQN(Trainer):
    def __init__(
        self,
        qf,
        policy,
        use_hard_updates=False,
        hard_update_period=1000,
        soft_update_tau=0.001,
        qf_lr=1e-3,
        qf_criterion=None,
        reward_scale=1.0,
        discount=0.99,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        **kwargs,
    ):
        """

        :param env: Env.
        :param qf: QFunction. Maps from state to action Q-values.
        :param learning_rate: Learning rate for qf. Adam is used.
        :param use_hard_updates: Use a hard rather than soft update.
        :param hard_update_period: How many gradient steps before copying the
        parameters over. Used if `use_hard_updates` is True.
        :param tau: Soft target tau to update target QF. Used if
        `use_hard_updates` is False.
        :param epsilon: Probability of taking a random action.
        :param kwargs: kwargs to pass onto TorchRLAlgorithm
        """
        self.policy = policy
        self.qf = qf
        self.target_qf = self.qf.copy()
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.soft_update_tau = soft_update_tau
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.reward_scale = reward_scale
        self.discount = discount

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )

        self.eval_statistics = None

    def train_step(self, batch):
        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Compute loss
        """
        target_q_values = self.target_qf(next_obs).detach().max(1, keepdim=True)[0]
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    ptu.get_numpy(y_pred),
                )
            )

    def _update_target_network(self):
        if self.use_hard_updates:
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
        else:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_update_tau)

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
            policy=self.policy,
        )

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
