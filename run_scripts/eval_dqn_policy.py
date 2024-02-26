import argparse
import inspect
import os
import pickle
import sys
from pathlib import Path

import gym
import joblib
import numpy as np
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print("sys.path:\n",sys.path)

import marlkit.torch.utils.pytorch_util as ptu
from marlkit.core import eval_util
from marlkit.envs import get_env, get_envs
from marlkit.launchers.launcher_util import set_seed
from marlkit.samplers import PathSampler
from marlkit.torch.common.policies import MakeDeterministic
from marlkit.core import PrefixDict
from marlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv

from video import save_video


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    #print("Obs Space: {}".format(env.observation_space))
    #print("Act Space: {}\n\n".format(env.action_space))
    model_data = joblib.load(variant["policy_checkpoint"])
    #policy_0 = model_data["policy_0"]
    #policy_1 = model_data["policy_1"]
    dict_cls = PrefixDict if variant["use_prefix_dict"] else dict
    policy_mapping_dict = dict_cls(
            zip(env.agent_ids, [f"policy_{i}" for i in range(env.n_agents)])
        )

    policy_trainer_n = {}
    policy_n = {}

    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_trainer_n:
            print(f"Load {policy_id} for {agent_id} ...")
            policy_n[policy_id] = model_data[f"{policy_id}"]["policy"]
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
    average_returns, std_returns = eval_util.get_agent_mean_avg_returns(test_paths, std=True)
    print(average_returns, std_returns)

    if variant["render"] and variant["render_mode"] == "rgb_array":
        video_path = variant["video_path"]
        video_path = os.path.join(video_path, variant["env_specs"]["env_name"])

        print("saving videos...")
        for i, test_path in enumerate(test_paths):
            images = np.stack(test_path["image"], axis=0)
            fps = 1 // getattr(env, "dt", 1 / 30)
            video_save_path = os.path.join(video_path, f"episode_{i}.mp4")
            save_video(images, video_save_path, fps=fps)

    return average_returns, std_returns, test_paths


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    parser.add_argument(
        "-s", "--save_res", help="save result to file", type=int, default=1
    )

    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
        #print("exp_specs: \n", exp_specs["constants"]['env_specs'])

    #exp_id = exp_specs["exp_id"]
    #exp_prefix = exp_specs["exp_name"]

    #seed = exp_specs["seed"]
    #set_seed(seed)
    # setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)
    average_returns, std_returns, test_paths = experiment(exp_specs['constants'])

    """train_file = (
        exp_specs["method"] + "-" + exp_specs["env_specs"]["env_name"] + "-STANDARD-EXP"
    )
    pkl_name = "/best.pkl"

    if "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
        pkl_name = "/params.pkl"

    if "gail" in exp_specs["method"]:
        if "hopper" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-hopper--rs-2.0"
        elif "walker" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-walker--rs-2.0"

    train_files = [train_file]
    save_path = "./final_performance/"

    for train_file in train_files:
        res_files = os.listdir("./logs/" + train_file)
        test_paths_all = []
        for file_ in res_files:
            exp_specs["policy_checkpoint"] = (
                "./logs/" + train_file + "/" + file_ + pkl_name
            )
            average_returns, std_returns, test_paths = experiment(exp_specs)
            test_paths_all.extend(test_paths)

            if args.save_res:
                save_dir = Path(save_path + train_file)
                save_dir.mkdir(exist_ok=True, parents=True)
                file_dir = save_dir.joinpath(
                    exp_specs["method"], exp_specs["env_specs"]["env_name"]
                )
                file_dir.mkdir(exist_ok=True, parents=True)

                if not os.path.exists(file_dir.joinpath("res.csv")):
                    with open(
                        save_dir.joinpath(
                            exp_specs["method"],
                            exp_specs["env_specs"]["env_name"],
                            "res.csv",
                        ),
                        "w",
                    ) as f:
                        f.write("avg,std\n")
                with open(
                    save_dir.joinpath(
                        exp_specs["method"],
                        exp_specs["env_specs"]["env_name"],
                        "res.csv",
                    ),
                    "a",
                ) as f:
                    f.write("{},{}\n".format(average_returns, std_returns))
        if exp_specs["save_samples"]:
            with open(
                Path(save_path).joinpath(
                    exp_specs["method"],
                    exp_specs["env_specs"]["env_name"],
                    "samples.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(test_paths_all, f)"""

