from typing import Dict, Any

import numpy as np

from marlkit.data_management.path_builder import PathBuilder


def rollout(
    env,
    eval_env,
    policy_n: Dict,
    policy_mapping_dict,
    max_path_length,
    no_terminal: bool = False,
    render: bool = False,
    render_kwargs: Dict[str, Any] = {},
):
    env_num = len(eval_env)
    path_builder = [PathBuilder() for _ in range(env_num)]

    ready_env_ids = np.arange(env_num)
    observations_n = eval_env.reset(ready_env_ids)
    env_num_steps = np.zeros(env_num, dtype=int)

    images = []
    for _ in range(max_path_length):
        actions_n = []
        for obs_n in observations_n:
            act_n = {}
            for a_id, obs in obs_n.items():
                p_id = policy_mapping_dict[a_id]
                act_n[a_id] = policy_n[p_id].get_actions(obs[None])[0]
            actions_n.append(act_n)
        if render:
            img = eval_env.render(**render_kwargs)
            images.append(img)

        next_observations_n, rewards_n, terminals_n, env_infos_n = eval_env.step(
            actions_n, ready_env_ids
        )
        if no_terminal:
            terminals_n = np.array(
                [{a_id: False for a_id in term_n.keys()} for term_n in terminals_n]
            )
        env_num_steps[ready_env_ids] += 1

        for idx, (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            terminal_n,
            env_info_n,
        ) in enumerate(
            zip(
                observations_n,
                actions_n,
                rewards_n,
                next_observations_n,
                terminals_n,
                env_infos_n,
            )
        ):
            env_idx = ready_env_ids[idx]
            for a_id in ob_n.keys():
                if a_id not in next_ob_n:
                    continue
                path_builder[env_idx][a_id].add_all(
                    observations=ob_n[a_id],
                    actions=action_n[a_id],
                    rewards=reward_n[a_id],
                    next_observations=next_ob_n[a_id],
                    terminals=terminal_n[a_id],
                    absorbings=np.array([0.0, 0.0]),
                    env_infos=env_info_n[a_id],
                )

        terminals_all = np.array([np.all(list(term_n.values())) for term_n in terminals_n])
        if np.any(terminals_all):
            end_env_ids = ready_env_ids[np.where(terminals_all)[0]]
            ready_env_ids = np.array(list(set(ready_env_ids) - set(end_env_ids)))
            if len(ready_env_ids) == 0:
                break

        observations_n = next_observations_n[np.where(terminals_all == False)]

    return path_builder, env_num_steps


class PathSampler:
    def __init__(
        self,
        env,
        eval_env,
        policy_n,
        policy_mapping_dict,
        num_steps,
        max_path_length,
        no_terminal=False,
        render=False,
        render_kwargs={},
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.eval_env = eval_env
        self.eval_env_num = len(eval_env)
        self.policy_n = policy_n
        self.policy_mapping_dict = policy_mapping_dict
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps

        while total_steps < num_steps:
            new_paths, path_lengths = rollout(
                self.env,
                self.eval_env,
                self.policy_n,
                self.policy_mapping_dict,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
                render_kwargs=self.render_kwargs,
            )
            paths.extend(new_paths)
            total_steps += path_lengths.sum()
        return paths
