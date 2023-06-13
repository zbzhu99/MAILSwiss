import argparse
import json
import os
import pickle
from pathlib import Path

import gym
import joblib
import numpy as np
import yaml

import marlkit.torch.utils.pytorch_util as ptu
from marlkit.core import eval_util
from marlkit.envs import get_env, get_envs
from marlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv
from marlkit.launchers.launcher_util import set_seed
from marlkit.samplers import PathSampler
from marlkit.torch.common.policies import MakeDeterministic


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    print("\n\nEnv: {}:{}".format(env_specs["env_creator"], env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(obs_space_n))
    print("Act Space: {}\n\n".format(act_space_n))

    with open(os.path.join(variant["policy_log_dir"], "variant.json"), "r") as f:
        policy_exp_specs = json.load(f)

    if policy_exp_specs.get("share_policy", True):
        policy_mapping_dict = dict(
            zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
        )
    else:
        policy_mapping_dict = dict(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_n = {}
    model_data = joblib.load(os.path.join(variant["policy_log_dir"], "params.pkl"))
    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_n:
            print(f"Load {policy_id} for {agent_id} ...")
            policy_n[policy_id] = model_data[policy_id]["policy"]
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    if variant["eval_deterministic"]:
        policy_n = {pid: MakeDeterministic(policy) for pid, policy in policy_n.items()}
    for policy in policy_n.values():
        policy.to(ptu.device)

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
    print("Average Returns: ", average_returns)
    print("Average Length: ", average_length)

    # if variant["render"] and variant["render_mode"] == "rgb_array":
    #     video_path = variant["video_path"]
    #     video_path = os.path.join(video_path, variant["env_specs"]["env_name"])

    #     print("saving videos...")
    #     for i, test_path in enumerate(test_paths):
    #         images = np.stack(test_path["image"], axis=0)
    #         fps = 1 // getattr(env, "dt", 1 / 30)
    #         video_save_path = os.path.join(video_path, f"episode_{i}.mp4")
    #         save_video(images, video_save_path, fps=fps)

    return average_returns, test_paths


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
