import argparse
import inspect
import os
import pickle
import random
import sys

import numpy as np
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym

import marlkit.torch.utils.pytorch_util as ptu
from marlkit.data_management.env_replay_buffer import EnvReplayBuffer
from marlkit.envs import get_env, get_envs
from marlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv
from marlkit.launchers.launcher_util import set_seed, setup_logger
from marlkit.torch.algorithms.bc.bc import BC
from marlkit.torch.common.policies import (
    DiscretePolicy,
    ReparamTanhMultivariateGaussianPolicy,
)


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.SafeLoader)

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    # PKL input format
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    if variant["traj_num"] != -1:
        traj_list = random.sample(traj_list, variant["traj_num"])

    env_specs = variant["env_specs"]
    env = get_env(env_specs)

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    expert_replay_buffer = EnvReplayBuffer(
        variant["bc_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(traj_list[i], env=env)

    if variant.get("share_policy", True):
        policy_mapping_dict = dict(
            zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
        )
    else:
        policy_mapping_dict = dict(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_n = {}

    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = env.observation_space_n[agent_id]
            act_space = env.action_space_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert len(obs_space.shape) == 1

            obs_dim = obs_space.shape[0]

            # build the policy models
            net_size = variant["policy_net_size"]
            num_hidden = variant["policy_num_hidden_layers"]
            if isinstance(act_space, gym.spaces.Box):
                action_dim = act_space.shape[0]
                policy = ReparamTanhMultivariateGaussianPolicy(
                    hidden_sizes=num_hidden * [net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )
            elif isinstance(act_space, gym.spaces.Discrete):
                action_dim = act_space.n
                policy = DiscretePolicy(
                    hidden_sizes=num_hidden * [net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )
            else:
                raise NotImplementedError("unknown action space: ", type(act_space))

            policy_n[policy_id] = policy
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    act_space_n = env.action_space_n
    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    env = env_wrapper(env)

    print("Creating {} training environments ...".format(env_specs["training_env_num"]))
    training_env = get_envs(
        env_specs, env_wrapper, env_num=env_specs["training_env_num"]
    )
    training_env.seed(env_specs["training_env_seed"])

    print("Creating {} evaluation environments ...".format(env_specs["eval_env_num"]))
    eval_env = get_envs(env_specs, env_wrapper, env_num=env_specs["eval_env_num"])
    eval_env.seed(env_specs["eval_env_seed"])

    algorithm = BC(
        env=env,
        training_env=training_env,
        eval_env=eval_env,
        exploration_policy_n=policy_n,
        expert_replay_buffer=expert_replay_buffer,
        policy_mapping_dict=policy_mapping_dict,
        **variant["bc_params"],
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    exp_suffix = "--lr-{}--trajnum-{}".format(
        exp_specs["bc_params"]["lr"],
        exp_specs["traj_num"],
    )

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    exp_prefix = exp_prefix + exp_suffix
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
