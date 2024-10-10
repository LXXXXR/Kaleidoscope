"""Runner for off-policy MA algorithms"""

import copy
from collections import deque
from statistics import mean

import wandb
import torch
import numpy as np

from harl.runners.off_policy_base_runner import OffPolicyBaseRunner


class OffPolicyMARunner(OffPolicyBaseRunner):
    """Runner for off-policy MA algorithms."""

    def __init__(self, args, algo_args, env_args):
        super(OffPolicyMARunner, self).__init__(args, algo_args, env_args)

        self.last_log_t = 0
        self.train_log_interval = self.algo_args["train"]["log_interval"]

        if "Kalei" in self.args["algo"]:
            self.actor_loss_history = deque(
                maxlen=algo_args["Kalei"]["deque_len"] * self.num_agents
            )
            self.div_loss_history = deque(
                maxlen=algo_args["Kalei"]["deque_len"] * self.num_agents
            )
            self.reset_interval = algo_args["Kalei"]["reset_interval"]
            self.reset_ratio = algo_args["Kalei"]["reset_ratio"]
            self.div_coef = algo_args["Kalei"]["div_coef"]
            self.last_reset_t = 0

            self.c_reset_interval = algo_args["algo"]["ensemble_args"]["reset_interval"]
            self.c_last_reset_t = 0

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            if "Kalei" in self.args["algo"]:
                next_actions.append(
                    self.actor[agent_id].get_target_actions(
                        sp_next_obs[agent_id], agent_id
                    )
                )
            else:
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
        if "Kalei" in self.args["algo"]:
            # reset_flag
            if (
                self.cur_step - self.c_last_reset_t > self.c_reset_interval
                and self.algo_args["train"]["num_env_steps"] - self.cur_step
                > self.c_reset_interval
            ):
                c_reset_flag = True
                self.c_last_reset_t = self.cur_step
            else:
                c_reset_flag = False
            (
                critic_loss,
                critic_div_loss,
                critic_loss_all,
                critic_div_coef,
            ) = self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
                c_reset_flag=c_reset_flag,
            )
        else:
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        if (
            "Kalei" in self.args["algo"]
            and self.cur_step - self.last_log_t >= self.train_log_interval
        ):

            wandb.log(
                {
                    f"critic_loss_all": critic_loss_all,
                    "timestep": self.cur_step,
                }
            )

        if self.total_it % self.policy_freq == 0:
            # reset actors
            if (
                "Kalei" in self.args["algo"]
                and self.cur_step - self.last_reset_t > self.reset_interval
                and self.algo_args["train"]["num_env_steps"] - self.cur_step
                > self.reset_interval
            ):
                # this is okay because we assert the share_param is True and all actors points to the same instance
                self.actor[0].reset_all_masks_weights(self.reset_ratio)
                self.last_reset_t = self.cur_step

            # train actors
            # actions shape: (n_agents, batch_size, dim)
            # implement random shuffle following harl
            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(np.random.permutation(self.num_agents))

            for agent_id in agent_order:
                actions = copy.deepcopy(torch.tensor(sp_actions)).to(self.device)
                self.actor[agent_id].turn_on_grad()
                # train this agent
                if "Kalei" in self.args["algo"]:
                    actions[agent_id] = self.actor[agent_id].get_actions(
                        sp_obs[agent_id], False, agent_id
                    )
                else:
                    actions[agent_id] = self.actor[agent_id].get_actions(
                        sp_obs[agent_id], False
                    )
                actions_list = [a for a in actions]
                actions_t = torch.cat(actions_list, dim=-1)
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                actor_loss = -torch.mean(value_pred)
                if self.cur_step - self.last_log_t >= self.train_log_interval:
                    wandb.log(
                        {
                            f"agent_{agent_id}_actor_loss": actor_loss.item(),
                            "timestep": self.cur_step,
                        }
                    )

                if "Kalei" in self.args["algo"]:
                    div_loss = self.actor[agent_id].mask_diversity_loss()
                    self.actor_loss_history.append(actor_loss.item())
                    self.div_loss_history.append(div_loss.item())
                    if mean(self.div_loss_history) != 0:
                        div_coef = abs(
                            self.div_coef
                            * mean(self.actor_loss_history)
                            / mean(self.div_loss_history)
                        )
                    else:
                        div_coef = self.div_coef
                    loss = actor_loss + div_coef * div_loss
                    if self.cur_step - self.last_log_t >= self.train_log_interval:
                        wandb.log(
                            {
                                f"agent_{agent_id}_div_loss": div_loss.item(),
                                "timestep": self.cur_step,
                            }
                        )

                else:
                    loss = actor_loss

                self.actor[agent_id].actor_optimizer.zero_grad()
                loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
            if (
                "Kalei" in self.args["algo"]
                and self.cur_step - self.last_log_t >= self.train_log_interval
            ):
                self.last_log_t = self.cur_step
            # soft update
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
