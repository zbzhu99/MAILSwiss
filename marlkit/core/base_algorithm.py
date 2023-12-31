import abc
import time
from collections import OrderedDict
from typing import Dict

import gtimer as gt
import numpy as np

from marlkit.core import dict_list_to_list_dict, eval_util, logger
from marlkit.data_management.env_replay_buffer import EnvReplayBuffer
from marlkit.data_management.path_builder import PathBuilder
from marlkit.policies.base import ExplorationPolicy
from marlkit.samplers import PathSampler
from marlkit.torch.common.policies import MakeDeterministic


class BaseAlgorithm(metaclass=abc.ABCMeta):
    """
    base algorithm for single task setting
    can be used for RL or Learning from Demonstrations
    """

    def __init__(
        self,
        env,
        exploration_policy_n: Dict[str, ExplorationPolicy],
        training_env=None,
        eval_env=None,
        eval_policy_n=None,
        eval_sampler=None,
        eval_sampler_func=PathSampler,
        eval_sampler_kwargs={},
        custom_eval_func=None,
        policy_mapping_dict=None,
        num_epochs=100,
        num_steps_per_epoch=10000,
        num_steps_between_train_calls=1000,
        num_steps_per_eval=1000,
        max_path_length=1000,
        min_steps_before_training=0,
        replay_buffer=None,
        replay_buffer_size=10000,
        freq_saving=1,
        save_replay_buffer=False,
        save_environment=False,
        save_algorithm=False,
        save_best=False,
        save_epoch=False,
        save_best_starting_from_epoch=0,
        best_key="AgentMeanAverageReturn",  # higher is better
        objective="maximize",
        no_terminal=False,
        eval_no_terminal=False,
        wrap_absorbing=False,
        render=False,
        render_kwargs={},
        freq_log_visuals=1,
        eval_deterministic=False,
    ):
        self.env = env
        self.env_num = len(training_env)
        self.training_env = training_env
        self.eval_env = eval_env
        self.exploration_policy_n = exploration_policy_n
        self.n_agents = env.n_agents
        self.agent_ids = env.agent_ids
        self.policy_ids = list(exploration_policy_n.keys())
        assert policy_mapping_dict is not None, "Require specifing agent-policy mapping"
        self.policy_mapping_dict = policy_mapping_dict
        self.agent_ids = list(policy_mapping_dict.keys())

        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_between_train_calls = num_steps_between_train_calls
        self.num_steps_per_eval = num_steps_per_eval
        self.max_path_length = max_path_length
        self.min_steps_before_training = min_steps_before_training

        self.render = render

        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.save_best = save_best
        self.save_epoch = save_epoch
        self.save_best_starting_from_epoch = save_best_starting_from_epoch
        self.best_key = best_key
        self.objective = objective
        if objective == "maximize":
            self.best_statistic_so_far = float("-Inf")
        elif objective == "minimize":
            self.best_statistic_so_far = float("Inf")
        else:
            raise NotImplementedError(objective)

        if eval_sampler is None:
            if eval_policy_n is None:
                eval_policy_n = exploration_policy_n
            eval_policy_n = {
                pid: MakeDeterministic(policy) for pid, policy in eval_policy_n.items()
            }
            eval_sampler = eval_sampler_func(
                env,
                eval_env,
                eval_policy_n,
                policy_mapping_dict,
                num_steps_per_eval,
                max_path_length,
                no_terminal=eval_no_terminal,
                render=render,
                render_kwargs=render_kwargs,
                **eval_sampler_kwargs,
            )
        self.eval_policy_n = eval_policy_n
        self.eval_sampler = eval_sampler

        self.action_space_n = env.action_space_n
        self.obs_space_n = env.observation_space_n
        self.replay_buffer_size = replay_buffer_size
        if replay_buffer is None:
            assert max_path_length < replay_buffer_size
            replay_buffer = EnvReplayBuffer(
                self.replay_buffer_size, self.env, random_seed=np.random.randint(10000)
            )
        else:
            assert max_path_length < replay_buffer._max_replay_buffer_size
        self.replay_buffer = replay_buffer

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._n_prev_train_env_steps = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = [
            PathBuilder(self.agent_ids) for _ in range(self.env_num)
        ]
        self._exploration_paths = []

        if wrap_absorbing:
            # needs to be properly handled both here and in replay buffer
            raise NotImplementedError()
        self.wrap_absorbing = wrap_absorbing
        self.freq_saving = freq_saving
        self.no_terminal = no_terminal

        self.eval_statistics = None
        self.freq_log_visuals = freq_log_visuals

        self.ready_env_ids = np.arange(self.env_num)

        if hasattr(self.env, "eval_paths"):
            self.custom_eval_func = self.env.eval_paths
        else:
            self.custom_eval_func = None

    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        # self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch)

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def start_training(self, start_epoch=0):
        self.ready_env_ids = np.arange(self.env_num)
        observations_n = self._start_new_rollout(
            self.ready_env_ids
        )  # Do it for support vec env

        self._current_path_builder = [
            PathBuilder(self.agent_ids) for _ in range(len(self.ready_env_ids))
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            total_rews_n = dict(
                zip(
                    self.agent_ids,
                    [
                        np.array([0.0 for _ in range(len(self.ready_env_ids))])
                        for _ in range(self.n_agents)
                    ],
                )
            )
            for steps_this_epoch in range(self.num_env_steps_per_epoch // self.env_num):
                actions_n = self._get_action_and_info(observations_n)

                for a_id in self.agent_ids:
                    if type(actions_n[a_id]) is tuple:
                        actions_n[a_id] = actions_n[a_id][0]

                if self.render:
                    self.training_env.render()

                (
                    next_obs_n,
                    raw_rewards_n,
                    terminals_n,
                    env_infos_n,
                ) = self.training_env.step(actions_n, self.ready_env_ids)
                if self.no_terminal:
                    terminals_n = dict(
                        zip(
                            self.agent_ids,
                            [
                                [False for _ in range(len(self.ready_env_ids))]
                                for _ in range(self.n_agents)
                            ],
                        )
                    )
                self._n_env_steps_total += len(self.ready_env_ids)

                rewards_n = raw_rewards_n
                for a_id in self.agent_ids:
                    total_rews_n[a_id] += raw_rewards_n[a_id]

                self._handle_vec_step(
                    observations_n,
                    actions_n,
                    rewards_n,
                    next_obs_n,
                    terminals_n,
                    absorbings_n={
                        a_id: [
                            np.array([0.0, 0.0]) for _ in range(len(self.ready_env_ids))
                        ]
                        for a_id in self.agent_ids
                    },
                    env_infos_n=env_infos_n,
                )

                terminals_all = np.ones_like(list(terminals_n.values())[0])
                for terminals in terminals_n.values():
                    terminals_all = np.logical_and(terminals_all, terminals)

                if np.any(terminals_all):
                    env_ind_local = np.where(terminals_all)[0]
                    for a_id in self.agent_ids:
                        total_rews_n[a_id][env_ind_local] = 0.0
                    if self.wrap_absorbing:
                        # raise NotImplementedError()
                        """
                        If we wrap absorbing states, two additional
                        transitions must be added: (s_T, s_abs) and
                        (s_abs, s_abs). In Disc Actor Critic paper
                        they make s_abs be a vector of 0s with last
                        dim set to 1. Here we are going to add the following:
                        ([next_ob,0], random_action, [next_ob, 1]) and
                        ([next_ob,1], random_action, [next_ob, 1])
                        This way we can handle varying types of terminal states.
                        """
                        # next_ob is the absorbing state
                        # for now just taking the previous action
                        self._handle_vec_step(
                            next_obs_n,
                            actions_n,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            rewards_n,
                            next_obs_n,
                            {
                                a_id: [
                                    np.array(
                                        [False for _ in range(len(self.ready_env_ids))]
                                    )
                                ]
                                for a_id in self.agent_ids
                            },
                            absorbings_n={
                                a_id: [
                                    np.array([0.0, 1.0])
                                    for _ in range(len(self.ready_env_ids))
                                ]
                                for a_id in self.agent_ids
                            },
                            env_infos_n=env_infos_n,
                        )
                        self._handle_vec_step(
                            next_obs_n,
                            actions_n,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            rewards_n,
                            next_obs_n,
                            {
                                a_id: [
                                    np.array(
                                        [False for _ in range(len(self.ready_env_ids))]
                                    )
                                ]
                                for a_id in self.agent_ids
                            },
                            absorbings_n={
                                a_id: [
                                    np.array([1.0, 1.0])
                                    for _ in range(len(self.ready_env_ids))
                                ]
                                for a_id in self.agent_ids
                            },
                            env_infos_n=env_infos_n,
                        )
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations_n = self._start_new_rollout(env_ind_local)
                    for a_id in self.agent_ids:
                        next_obs_n[a_id][env_ind_local] = reset_observations_n[a_id]
                if np.any(
                    np.array(
                        [len(self._current_path_builder[i]) for i in self.ready_env_ids]
                    )
                    >= self.max_path_length
                ):
                    env_ind_local = np.where(
                        np.array(
                            [
                                len(self._current_path_builder[i])
                                for i in self.ready_env_ids
                            ]
                        )
                        >= self.max_path_length
                    )[0]
                    self._handle_vec_rollout_ending(env_ind_local)
                    reset_observations_n = self._start_new_rollout(env_ind_local)
                    for a_id in self.agent_ids:
                        next_obs_n[a_id][env_ind_local] = reset_observations_n[a_id]

                observations_n = next_obs_n

                if (
                    self._n_env_steps_total - self._n_prev_train_env_steps
                ) >= self.num_steps_between_train_calls:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def _try_to_train(self, epoch):
        if self._can_train():
            self._n_prev_train_env_steps = self._n_env_steps_total
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        if self._can_evaluate():
            # save if it's time to save
            if (int(epoch) % self.freq_saving == 0) or (epoch + 1 >= self.num_epochs):
                # if epoch + 1 >= self.num_epochs:
                # epoch = 'final'
                logger.save_extra_data(self.get_extra_data_to_save(epoch))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

            self.evaluate(epoch)

            logger.record_tabular(
                "Number of train calls total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs["train"][-1]
            sample_time = times_itrs["sample"][-1]
            eval_time = times_itrs["eval"][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular("Train Time (s)", train_time)
            logger.record_tabular("(Previous) Eval Time (s)", eval_time)
            logger.record_tabular("Sample Time (s)", sample_time)
            logger.record_tabular("Epoch Time (s)", epoch_time)
            logger.record_tabular("Total Train Time (s)", total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return len(self._exploration_paths) > 0 and self._n_prev_train_env_steps > 0

    def _can_train(self):
        return (
            self.replay_buffer.num_steps_can_sample() >= self.min_steps_before_training
        )

    def _get_action_and_info(self, observation_n):
        """
        Get an action to take in the environment.
        :param observation_n:
        :return:
        """
        action_n = {}
        for agent_id, observation in observation_n.items():
            policy_id = self.policy_mapping_dict[agent_id]
            self.exploration_policy_n[policy_id].set_num_steps_total(
                self._n_env_steps_total
            )
            action_n[agent_id] = self.exploration_policy_n[policy_id].get_actions(
                observation
            )
        return action_n

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix("Iteration #%d | " % epoch)

    def _end_epoch(self):
        self.eval_statistics = None
        logger.log("Epoch Duration: {0}".format(time.time() - self._epoch_start_time))
        logger.log("Started Training: {0}".format(self._can_evaluate()))
        logger.pop_prefix()

    def _start_new_rollout(self, env_ind_local):
        # self.exploration_policy.reset() # Do nothing originally at all
        self.env_ind_global = self.ready_env_ids[env_ind_local]
        return self.training_env.reset(env_ind_local)

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            terminal_n,
            absorbing_n,
            env_info_n,
        ) in zip(
            *map(
                dict_list_to_list_dict,
                [
                    path.get_all_agent_dict("observations"),
                    path.get_all_agent_dict("actions"),
                    path.get_all_agent_dict("rewards"),
                    path.get_all_agent_dict("next_observations"),
                    path.get_all_agent_dict("terminals"),
                    path.get_all_agent_dict("absorbings"),
                    path.get_all_agent_dict("env_infos"),
                ],
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                terminal_n,
                absorbing_n,
                env_info_n=env_info_n,
                path_builder=False,
            )

    def _handle_vec_step(
        self,
        observations_n: Dict,
        actions_n: Dict,
        rewards_n: Dict,
        next_observations_n: Dict,
        terminals_n: Dict,
        absorbings_n: Dict,
        env_infos_n: Dict,
    ):
        """
        Implement anything that needs to happen after every step under vec envs
        :return:
        """
        for env_idx, (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            terminal_n,
            absorbing_n,
            env_info_n,
        ) in enumerate(
            zip(
                *map(
                    dict_list_to_list_dict,
                    [
                        observations_n,
                        actions_n,
                        rewards_n,
                        next_observations_n,
                        terminals_n,
                        absorbings_n,
                        env_infos_n,
                    ],
                )
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                terminal_n,
                absorbing_n=absorbing_n,
                env_info_n=env_info_n,
                env_idx=env_idx,
                add_buf=False,
            )

    def _handle_step(
        self,
        observation_n,
        action_n,
        reward_n,
        next_observation_n,
        terminal_n,
        absorbing_n,
        env_info_n,
        env_idx=0,
        add_buf=True,
        path_builder=True,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        if path_builder:
            for a_id in self.agent_ids:
                self._current_path_builder[env_idx][a_id].add_all(
                    observations=observation_n[a_id],
                    actions=action_n[a_id],
                    rewards=reward_n[a_id],
                    next_observations=next_observation_n[a_id],
                    terminals=terminal_n[a_id],
                    absorbings=absorbing_n[a_id],
                    env_infos=env_info_n[a_id],
                )
        if add_buf:
            self.replay_buffer.add_sample(
                observation_n=observation_n,
                action_n=action_n,
                reward_n=reward_n,
                terminal_n=terminal_n,
                next_observation_n=next_observation_n,
                absorbing_n=absorbing_n,
                env_info_n=env_info_n,
            )

    def _handle_vec_rollout_ending(self, end_idx):
        """
        Implement anything that needs to happen after every vec env rollout.
        """
        for idx in end_idx:
            self._handle_path(self._current_path_builder[idx])
            self.replay_buffer.terminate_episode()
            self._n_rollouts_total += 1
            if len(self._current_path_builder[idx]) > 0:
                self._exploration_paths.append(self._current_path_builder[idx])
                self._current_path_builder[idx] = PathBuilder(self.agent_ids)

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode()
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(self._current_path_builder)
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        """
        Probably will be overridden by each algorithm
        """
        data_to_save = dict(
            epoch=epoch,
            exploration_policy_n=self.exploration_policy_n,
        )
        if self.save_environment:
            data_to_save["env"] = self.training_env
        return data_to_save

    # @abc.abstractmethod
    # def load_snapshot(self, snapshot):
    #     """
    #     Should be implemented on a per algorithm basis
    #     taking into consideration the particular
    #     get_epoch_snapshot implementation for the algorithm
    #     """
    #     pass

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save["env"] = self.training_env
        if self.save_replay_buffer:
            data_to_save["replay_buffer"] = self.replay_buffer
        if self.save_algorithm:
            data_to_save["algorithm"] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except TypeError:
            print("No Stats to Eval")

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(
            eval_util.get_generic_path_information(
                test_paths,
                stat_prefix="Test",
            )
        )
        if self.custom_eval_func:
            statistics.update(
                self.custom_eval_func(
                    test_paths,
                    stat_prefix="Test",
                )
            )

        if len(self._exploration_paths) > 0:
            statistics.update(
                eval_util.get_generic_path_information(
                    self._exploration_paths,
                    stat_prefix="Exploration",
                )
            )
            if self.custom_eval_func:
                statistics.update(
                    self.custom_eval_func(
                        self._exploration_paths,
                        stat_prefix="Exploration",
                    )
                )

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if int(epoch) % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())

        agent_mean_avg_returns = eval_util.get_agent_mean_avg_returns(test_paths)
        statistics["AgentMeanAverageReturn"] = agent_mean_avg_returns
        for key, value in statistics.items():
            logger.record_tabular(key, np.mean(value))

        best_statistic = statistics[self.best_key]
        data_to_save = {"epoch": epoch, "statistics": statistics}
        data_to_save.update(self.get_epoch_snapshot(epoch))
        if self.save_epoch:
            logger.save_extra_data(data_to_save, "epoch{}.pkl".format(epoch))
            print("\n\nSAVED MODEL AT EPOCH {}\n\n".format(epoch))
        if (
            self.objective == "maximize"
            and best_statistic > self.best_statistic_so_far
            or self.objective == "minimize"
            and best_statistic < self.best_statistic_so_far
        ):
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {"epoch": epoch, "statistics": statistics}
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, "best.pkl")
                print("\n\nSAVED BEST\n\n")
