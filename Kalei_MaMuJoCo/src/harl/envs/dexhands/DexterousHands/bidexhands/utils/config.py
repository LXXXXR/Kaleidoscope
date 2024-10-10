# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

import argparse


def set_np_formatting():
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def warn_task_name():
    raise Exception("Unrecognized task!")


def warn_algorithm_name():
    raise Exception(
        "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]"
    )


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args, use_rlg_config=False):
    if args.task in [
        "ShadowHandOver",
        "ShadowHandCatchUnderarm",
        "ShadowHandTwoCatchUnderarm",
        "ShadowHandCatchAbreast",
        "ShadowHandReOrientation",
        "ShadowHandCatchOver2Underarm",
        "ShadowHandBottleCap",
        "ShadowHandDoorCloseInward",
        "ShadowHandDoorCloseOutward",
        "ShadowHandDoorOpenInward",
        "ShadowHandDoorOpenOutward",
        "ShadowHandKettle",
        "ShadowHandPen",
        "ShadowHandSwitch",
        "ShadowHandPushBlock",
        "ShadowHandSwingCup",
        "ShadowHandGraspAndPlace",
        "ShadowHandScissors",
        "AllegroHandOver",
        "AllegroHandCatchUnderarm",
    ]:
        return "cfg/{}.yaml".format(args.task)

    elif args.task in ["ShadowHandLiftUnderarm"]:
        return "cfg/{}.yaml".format(args.task)

    elif args.task in ["ShadowHandBlockStack"]:
        return "cfg/{}.yaml".format(args.task)

    elif args.task in ["ShadowHand", "ShadowHandReOrientation"]:
        return "cfg/{}.yaml".format(args.task)

    else:
        warn_task_name()


def load_env_cfg(args, use_rlg_config=False):
    with open(
        os.path.join(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], args.cfg_env
        ),
        "r",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    return cfg


def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(env_args, benchmark=False, use_rlg_config=False):
    custom_parameters = [
        {"name": "--env", "type": str, "default": "dexhands", "help": "env to run"},
        {
            "name": "--exp_name",
            "type": str,
            "default": "installtest",
            "help": "experiment name",
        },
        {
            "name": "--test",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, no training",
        },
        {
            "name": "--play",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library",
        },
        {
            "name": "--resume",
            "type": int,
            "default": 0,
            "help": "Resume training or start testing from a checkpoint",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library",
        },
        {
            "name": "--task",
            "type": str,
            "default": env_args["task"],
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity",
        },
        {
            "name": "--task_type",
            "type": str,
            "default": "Python",
            "help": "Choose Python or C++",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network",
        },
        {"name": "--logdir", "type": str, "default": "logs/"},
        {
            "name": "--experiment",
            "type": str,
            "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name",
        },
        {
            "name": "--metadata",
            "action": "store_true",
            "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user",
        },
        {"name": "--cfg_train", "type": str, "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {
            "name": "--num_envs",
            "type": int,
            "default": env_args["n_threads"],
            "help": "Number of environments to create - override config file",
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": env_args["hands_episode_length"],
            "help": "Episode length, by default is read from yaml config",
        },
        {"name": "--seed", "type": int, "help": "Random seed"},
        {
            "name": "--max_iterations",
            "type": int,
            "default": 0,
            "help": "Set a maximum number of training iterations",
        },
        {
            "name": "--steps_num",
            "type": int,
            "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings.",
        },
        {
            "name": "--minibatch_size",
            "type": int,
            "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings.",
        },
        {
            "name": "--randomize",
            "action": "store_true",
            "default": False,
            "help": "Apply physics domain randomization",
        },
        {
            "name": "--torch_deterministic",
            "action": "store_true",
            "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour",
        },
        {
            "name": "--algo",
            "type": str,
            "default": "happo",
            "help": "Choose an algorithm",
        },
        {
            "name": "--model_dir",
            "type": str,
            "default": "",
            "help": "Choose a model dir",
        },
        {
            "name": "--datatype",
            "type": str,
            "default": "random",
            "help": "Choose an ffline datatype",
        },
    ]

    if benchmark:
        custom_parameters += [
            {
                "name": "--num_proc",
                "type": int,
                "default": 1,
                "help": "Number of child processes to launch",
            },
            {
                "name": "--random_actions",
                "action": "store_true",
                "help": "Run benchmark with random actions instead of inferencing",
            },
            {
                "name": "--bench_len",
                "type": int,
                "default": 10,
                "help": "Number of timing reports",
            },
            {
                "name": "--bench_file",
                "action": "store",
                "help": "Filename to store benchmark results",
            },
        ]

    parser = argparse.ArgumentParser()
    args, unknown_args = parser.parse_known_args()
    for i in range(0, len(unknown_args), 2):
        exist = False
        for param in custom_parameters:
            if param["name"] == unknown_args[i]:
                exist = True
                break
        if exist:
            continue
        else:
            custom_parameters += [{"name": unknown_args[i], "type": str}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else "cpu"

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    cfg_env = retrieve_cfg(args, use_rlg_config)

    if use_rlg_config == False:
        if args.horovod:
            print(
                "Distributed multi-gpu training with Horovod is not supported by rl-pytorch. Use rl_games for distributed training."
            )
        if args.steps_num != -1:
            print(
                "Setting number of simulation steps per iteration from command line is not supported by rl-pytorch."
            )
        if args.minibatch_size != -1:
            print(
                "Setting minibatch size from command line is not supported by rl-pytorch."
            )
        if args.checkpoint != "Base":
            raise ValueError(
                "--checkpoint is not supported by rl-pytorch. Please use --resume <iteration number>"
            )

    # use custom parameters if provided by user

    if args.cfg_env == "Base":
        args.cfg_env = cfg_env

    # if args.algo not in ["maddpg", "happo", "mappo", "hatrpo","ippo","ppo","sac","td3","ddpg","trpo"]:
    #     warn_algorithm_name()

    return args
