import gym
from gym.spaces import Tuple
from pretrained.ddpg import DDPG, EpymarlDDPG
import torch
import os

import numpy as np


class PretrainedWorld_4v2(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-2])
        self.observation_space = Tuple(self.observation_space[:-2])
        self.n_agents = 4

        self.prey1 = DDPG(32, 5, 50, 128, 0.01)
        self.prey2 = DDPG(32, 5, 50, 128, 0.01)
        param_path = os.path.join(
            os.path.dirname(__file__), "simple_world_4v2_maddpg.pt"
        )
        save_dict = torch.load(param_path)
        self.prey1.load_params(save_dict["agent_params"][-2])
        self.prey2.load_params(save_dict["agent_params"][-1])
        self.prey1.policy.eval()
        self.prey2.policy.eval()
        self.last_prey_obs = None

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = obs[-2:]
        return obs[:-2]

    def step(self, action):
        prey1_action = self.prey1.step(self.last_prey_obs[0])
        prey2_action = self.prey2.step(self.last_prey_obs[1])
        action = tuple(action) + (prey1_action, prey2_action)
        obs, rew, done, info = super().step(action)
        self.last_prey_obs = obs[-2:]
        obs = obs[:-2]
        rew = rew[:-2]
        done = done[:-2]
        return obs, rew, done, info
