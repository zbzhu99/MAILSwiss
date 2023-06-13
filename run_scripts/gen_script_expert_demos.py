import argparse
import inspect
import os
import pickle
import sys
from pathlib import Path

import gym
import numpy as np
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import marlkit.torch.utils.pytorch_util as ptu
from marlkit.core import eval_util
from marlkit.envs import get_env, get_envs
from marlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv
from marlkit.launchers.launcher_util import set_seed
from marlkit.samplers import PathSampler
from marlkit.scripted_experts import get_scripted_policy


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    print("\n\nEnv: {}:{}".format(env_specs["env_creator"], env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(obs_space_n))
    print("Act Space: {}\n\n".format(act_space_n))

    if variant["share_policy"]:
        policy_mapping_dict = dict(
            zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
        )
    else:
        policy_mapping_dict = dict(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_n = {}
    for agent_idx, agent_id in enumerate(env.agent_ids):
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_n:
            print(f"Create scripted {policy_id} for {agent_id} ...")
            policy_n[policy_id] = get_scripted_policy(
                variant["scripted_policy_name"], env, agent_idx
            )
        else:
            print(f"Use existing scripted {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    print("Creating {} evaluation environments ...".format(env_specs["eval_env_num"]))
    eval_env = get_envs(env_specs, env_wrapper, env_num=env_specs["eval_env_num"])
    eval_env.seed(env_specs["eval_env_seed"])

    eval_sampler = PathSampler(
        env,
        eval_env,
        policy_n,
        policy_mapping_dict,
        variant["num_eval_steps"],
        variant["max_path_length"],
        no_terminal=variant["no_terminal"],
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = eval_util.get_agent_mean_avg_returns(test_paths)
    average_length = np.mean([len(path) for path in test_paths])
    max_length = np.max([len(path) for path in test_paths])
    print("Average Returns: ", average_returns)
    print("Average Length: ", average_length)
    print("Max Length: ", max_length)

    return average_returns, test_paths


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)

    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]

    seed = exp_specs["seed"]
    set_seed(seed)

    average_returns, test_paths = experiment(exp_specs)

    save_dir = Path("./demos").joinpath(
        exp_specs["env_specs"]["env_creator"], exp_specs["env_specs"]["env_name"]
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(
        save_dir.joinpath(
            "expert_{}.pkl".format(exp_specs["method"]),
        ),
        "wb",
    ) as f:
        pickle.dump(test_paths, f)
