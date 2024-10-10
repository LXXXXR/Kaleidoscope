from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .n_controller import NMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np


# This multi-agent controller shares parameters between agents
class Kalei_type_NMAC(NMAC):
    def __init__(self, scheme, groups, args):
        super(Kalei_type_NMAC, self).__init__(scheme, groups, args)
        self.n_unit_types = args.n_unit_types
        assert (
            args.env_args["state_timestep_number"] is False
        ), "the unit type slcing is wrong if otherwise"

    @property
    def sparsities(self):
        return self.agent.get_sparsities()

    @property
    def mask_parameters(self):
        return self.agent.mask_parameters

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            qvals[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs, unit_ids = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        else:
            self.agent.train()
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, unit_ids
        )

        self.agent.train()
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        unit_ids_onehot = batch["obs"][:, t, :, -self.n_unit_types :]
        # the agent is either dead -> all zeros or alive -> one hot
        assert (
            th.logical_or(
                unit_ids_onehot.sum(dim=-1) == 1, unit_ids_onehot.sum(dim=-1) == 0
            )
        ).all(), "recheck codes for selecting unit types"
        unit_ids = th.argmax(unit_ids_onehot, dim=-1)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        unit_ids = unit_ids.reshape(bs, self.n_agents)
        return inputs, unit_ids
