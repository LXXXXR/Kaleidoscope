"""MATD3 algorithm."""

from copy import deepcopy

import torch

from harl.algorithms.actors.hatd3 import HATD3
from harl.models.policy_models.deterministic_policy import Kalei_DeterministicPolicy
from harl.utils.envs_tools import check


class MATD3(HATD3):
    pass


class Kalei_MATD3(MATD3):
    """MATD3 algorithm with Kaleidoscope parameter sharing."""

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]
        # sanity check, Kaleidoscope must be implemented under parameter sharing
        assert args["share_param"]

        self.actor = Kalei_DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

    @property
    def sparsities(self):
        return self.actor.get_sparsities()

    @property
    def mask_parameters(self):
        return self.actor.mask_parameters

    def mask_diversity_loss(self):
        return self.actor.mask_diversity_loss()

    def reset_all_masks_weights(self, reset_ratio):
        return self.actor._reset_all_masks_weights(reset_ratio)

    def get_actions(self, obs, add_noise, mask_id):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs, mask_id)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        return actions

    def get_target_actions(self, obs, mask_id):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.target_actor(obs, mask_id)
        noise = torch.randn_like(actions) * self.policy_noise * self.scale
        noise = torch.clamp(
            noise, -self.noise_clip * self.scale, self.noise_clip * self.scale
        )
        actions += noise
        actions = torch.clamp(actions, self.low, self.high)
        return actions
