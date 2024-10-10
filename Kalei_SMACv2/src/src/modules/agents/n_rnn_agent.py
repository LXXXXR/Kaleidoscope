import math
from math import ceil
from functools import partial


import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init


class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.apply(weights_init)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q.view(b, a, -1), h.view(b, a, -1)


class NRNNAgent_1R3(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent_1R3, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = F.relu(self.fc2(h), inplace=True)
        q = F.relu(self.fc3(q), inplace=True)
        q = self.fc4(q)
        return q.view(b, a, -1), h.view(b, a, -1)


class compare_STE(th.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return F.relu(th.sign(input))

    @staticmethod
    def backward(ctx, grad_output):
        return th.tanh_(grad_output)


class KaleiLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        n_masks,
        threshold_init_scale,
        threshold_init_bias,
        threshold_reset_scale,
        threshold_reset_bias,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, *args, **kwargs
        )
        self.n_masks = n_masks
        self.threshold_init_scale = threshold_init_scale
        self.threshold_init_bias = threshold_init_bias
        self.threshold_reset_scale = threshold_reset_scale
        self.threshold_reset_bias = threshold_reset_bias
        # initialize the thresholds as zeros first
        init_sacles = th.rand_like(self.weight[None, ...].repeat(n_masks, 1, 1))
        init_sacles = -(init_sacles * threshold_init_scale + threshold_init_bias)
        self.sparse_thresholds = nn.Parameter(init_sacles)
        self.activation = th.relu
        self.f = th.sigmoid

    def _sparse_function(self, w, mask_ids):
        # mask_ids with the dimension of 1 (bs, ), w is of dimension (out_dim, in_dim)
        all_mask_ids = th.arange(self.n_masks).to(self.weight.device)
        # (n_masks, out_dim, in_dim)
        thresholds = self.sparse_thresholds[all_mask_ids]
        t = self.f(thresholds)

        w = th.sign(w[None, ...]) * self.activation(th.abs(w[None, ...]) - t)
        # (bs, out_dim, in_dim)
        return w[mask_ids]

    def _get_masks(self, w, mask_ids):
        all_mask_ids = th.arange(self.n_masks).to(self.weight.device)
        # (n_masks, out_dim, in_dim)
        thresholds = self.sparse_thresholds[all_mask_ids]
        # very important thing is we need to make it differentiable
        w = compare_STE.apply(th.abs(w[None, ...].detach()) - self.f(thresholds))
        # (bs, out_dim, in_dim)
        return w[mask_ids]

    def _get_weighted_masks(self, w, mask_ids):
        masks = self._get_masks(w, mask_ids)
        return masks * th.abs(w[None, ...].detach())

    def forward(self, x, mask_ids):
        assert len(x.shape) == 2, f"Invalid input shape {x.shape}."
        w = self._sparse_function(self.weight, mask_ids)
        result = th.matmul(w, x[:, :, None]).squeeze(-1) + self.bias[None, :]
        return result

    def get_sparsities(self):
        mask_ids = th.arange(self.n_masks).to(self.weight.device)
        # (mask_ids, out_dim, in_dim)
        w = self._sparse_function(self.weight, mask_ids)
        temp = w.detach()
        temp[temp != 0] = 1
        temp = temp.reshape(self.n_masks, -1)
        return 1.0 - temp.mean(dim=-1), temp[0].numel()

    def _reset(self, reset_ratio):
        with th.no_grad():
            reset_flag = th.rand_like(self.weight)
            reset_flag = reset_flag < reset_ratio  # 1 means reset
            all_mask_ids = th.arange(self.n_masks).to(self.weight.device)
            # (n_masks, out_dim, in_dim)
            thresholds = self.sparse_thresholds[all_mask_ids]
            # very important thing is we need to make it differentiable
            m = compare_STE.apply(
                th.abs(self.weight[None, ...].detach()) - self.f(thresholds)
            )
            # (out_dim, in_dim)
            m = m.sum(dim=0) == 0  # 1 means all masks are zeros
            reset_flag = th.logical_and(reset_flag, m).float()

            # reinit the thresholds
            init_sacles = th.rand_like(
                self.weight[None, ...].repeat(self.n_masks, 1, 1)
            )
            init_sacles = -(
                init_sacles * self.threshold_reset_scale + self.threshold_reset_bias
            )
            new_sparse_thresholds = (
                reset_flag[None, ...] * init_sacles
                + (1 - reset_flag[None, ...]) * self.sparse_thresholds
            )
            self.sparse_thresholds.copy_(new_sparse_thresholds)

            new_weights = th.zeros_like(self.weight)
            init.kaiming_uniform_(new_weights, a=math.sqrt(5))
            new_weights = reset_flag * new_weights + (1 - reset_flag) * self.weight
            self.weight.copy_(new_weights)


class Kalei_type_NRNNAgent_1R3(nn.Module):
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_unit_types

        self.Kalei_linear = partial(
            KaleiLinear,
            n_masks=self.n_agents,
            threshold_init_scale=args.Kalei_args["threshold_init_scale"],
            threshold_init_bias=args.Kalei_args["threshold_init_bias"],
            threshold_reset_scale=args.Kalei_args["threshold_reset_scale"],
            threshold_reset_bias=args.Kalei_args["threshold_reset_bias"],
        )

        self.fc1 = self.Kalei_linear(
            in_features=input_shape, out_features=args.rnn_hidden_dim
        )
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = self.Kalei_linear(
            in_features=args.rnn_hidden_dim, out_features=args.rnn_hidden_dim
        )
        self.fc3 = self.Kalei_linear(
            in_features=args.rnn_hidden_dim, out_features=args.rnn_hidden_dim
        )
        self.fc4 = self.Kalei_linear(
            in_features=args.rnn_hidden_dim, out_features=args.n_actions
        )
        self.mask_layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.reset_layers = [self.fc2, self.fc3, self.fc4]
        if args.Kalei_args["sparsity_layer_weights"]:
            self.sparsity_layer_weights = args.Kalei_args["sparsity_layer_weights"]
        else:
            self.sparsity_layer_weights = [1.0 for _ in self.mask_layers]
        self.set_require_grads(mode=False)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, agent_ids):
        # agent dimention is indexed
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        agent_ids = agent_ids.reshape(-1)

        x = F.relu(self.fc1(inputs, agent_ids))

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = F.relu(self.fc2(h, agent_ids))
        q = F.relu(self.fc3(q, agent_ids))
        q = self.fc4(q, agent_ids)

        return q.view(b, a, -1), h.view(b, a, -1)

    @property
    def mask_parameters(self):
        return [l.sparse_thresholds for l in self.mask_layers]

    def set_require_grads(self, mode):
        assert mode in [True, False], f"Invalid mode {mode}."
        for l in self.mask_layers:
            l.sparse_thresholds.requires_grad = mode

    def _get_linear_weight_sparsities(self):
        sparsities = th.zeros(self.n_agents, len(self.mask_layers))
        w_counts = []
        for l in range(len(self.mask_layers)):
            sparsities[:, l], w_count = self.mask_layers[l].get_sparsities()
            w_counts.append(w_count)
        # if sparsity is 0.6, means 60% zeros and 40% non-zeros
        return sparsities, w_counts

    def get_sparsities(self):
        # calculate overall number of parameters first
        total_params = 0
        for n, p in self.named_parameters():
            if "sparse_thresholds" not in n:
                total_params += p.numel()

        # calculate overall number of zero parameters
        w_sparsities, w_counts = self._get_linear_weight_sparsities()
        # take average over the number of agents
        w_sparsities, w_sparsities_var = w_sparsities.mean(dim=0), w_sparsities.var(
            dim=0
        )
        zero_params = 0
        for l in range(len(self.mask_layers)):
            zero_params += w_counts[l] * w_sparsities[l]

        return w_sparsities, w_sparsities_var, (zero_params / total_params).item()

    def mask_diversity_loss(self):
        loss = 0
        mask_ids = th.arange(self.n_agents).to(self.fc1.weight.device)
        for l, l_w in zip(self.mask_layers, self.sparsity_layer_weights):
            # (n_masks, out_dim, in_dim)
            m = l._get_weighted_masks(l.weight, mask_ids)
            # every two masks distance
            loss += l_w * th.abs(m[None, :, :, :] - m[:, None, :, :]).sum()

        _, w_counts = self._get_linear_weight_sparsities()
        param_count = sum(w_counts)
        # we want to maximize the diversity
        # averaged over the number of agents and param_count to make it range from 0 to 1
        return -loss / (self.n_agents * (self.n_agents - 1)) / param_count

    def _reset_all_masks_weights(self, reset_ratio):
        if isinstance(reset_ratio, float):
            reset_ratio = [reset_ratio for _ in self.reset_layers]
        for l, r in zip(self.reset_layers, reset_ratio):
            l._reset(r)
