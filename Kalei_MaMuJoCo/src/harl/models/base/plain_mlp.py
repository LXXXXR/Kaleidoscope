import math
from functools import partial
from copy import deepcopy

import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from harl.utils.models_tools import get_active_func


class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


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
        **kwargs,
    ):
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
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

    def _sparse_function(self, w, mask_id):
        # mask_ids with the dimension of 1 (bs, ), w is of dimension (out_dim, in_dim)
        # (out_dim, in_dim)
        thresholds = self.sparse_thresholds[mask_id]
        t = self.f(thresholds)

        w = th.sign(w) * self.activation(th.abs(w) - t)
        # (out_dim, in_dim)
        return w

    def _get_masks(self, w, mask_ids):
        all_mask_ids = th.arange(self.n_masks).to(self.weight.device)
        # (n_masks, out_dim, in_dim)
        thresholds = self.sparse_thresholds[all_mask_ids]
        # very important thing is we need to make it differentiable
        w = compare_STE.apply(th.abs(w[None, ...].detach()) - self.f(thresholds))
        return w[mask_ids]

    def _get_weighted_masks(self, w, mask_ids):
        masks = self._get_masks(w, mask_ids)
        return masks * th.abs(w[None, ...].detach())

    def forward(self, x, mask_id):
        assert len(x.shape) == 2, f"Invalid input shape {x.shape}."
        # (out_dim, in_dim)
        w = self._sparse_function(self.weight, mask_id)
        result = F.linear(x, w, self.bias)
        return result

    def get_sparsities(self):
        # (mask_ids, out_dim, in_dim)
        temp = []
        for mask_id in range(self.n_masks):
            temp.append(self._sparse_function(self.weight, mask_id))
        temp = th.stack(temp, dim=0)
        # (mask_ids, out_dim, in_dim)
        temp = temp.detach()
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

    def _reset_mask(self, mask_id):
        with th.no_grad():
            new_threshold = th.rand_like(self.sparse_thresholds[mask_id])
            new_threshold = -(
                new_threshold * self.threshold_reset_scale + self.threshold_reset_bias
            )
            new_thresholds = deepcopy(self.sparse_thresholds)
            new_thresholds[mask_id] = new_threshold
            self.sparse_thresholds.copy_(new_thresholds)


class Kalei_MLP(nn.Module):

    def __init__(self, sizes, activation_func, args, final_activation_func="identity"):
        super().__init__()

        Kalei_args = args["Kalei_args"]
        self.n_masks = Kalei_args["n_masks"]

        self.Kalei_linear = partial(
            KaleiLinear,
            n_masks=Kalei_args["n_masks"],
            threshold_init_scale=Kalei_args["threshold_init_scale"],
            threshold_init_bias=Kalei_args["threshold_init_bias"],
            threshold_reset_scale=Kalei_args["threshold_reset_scale"],
            threshold_reset_bias=Kalei_args["threshold_reset_bias"],
        )

        # need to assert, not general implementation
        assert len(sizes) == 4

        self.activation_func = get_active_func(activation_func)
        self.final_activation_func = get_active_func(final_activation_func)
        self.fc1 = self.Kalei_linear(in_features=sizes[0], out_features=sizes[1])
        self.fc2 = self.Kalei_linear(in_features=sizes[1], out_features=sizes[2])
        self.fc3 = self.Kalei_linear(in_features=sizes[2], out_features=sizes[3])

        self.layer_norm1 = nn.LayerNorm(sizes[1])
        self.layer_norm2 = nn.LayerNorm(sizes[2])

        self.mask_layers = [self.fc1, self.fc2, self.fc3]
        self.reset_layers = [self.fc1, self.fc2, self.fc3]
        self.sparsity_layer_weights = Kalei_args["sparsity_layer_weights"]
        self.weighted_masks = Kalei_args["weighted_masks"]
        self.norm_flag = Kalei_args.get("norm_flag", False)

    def forward(self, x, mask_id):
        if self.norm_flag:
            x = self.activation_func(self.layer_norm1(self.fc1(x, mask_id)))
            x = self.activation_func(self.layer_norm2(self.fc2(x, mask_id)))
        else:
            x = self.activation_func(self.fc1(x, mask_id))
            x = self.activation_func(self.fc2(x, mask_id))

        x = self.final_activation_func(self.fc3(x, mask_id))

        return x

    @property
    def mask_parameters(self):
        return [l.sparse_thresholds for l in self.mask_layers]

    def _get_linear_weight_sparsities(self):
        sparsities = th.zeros(self.n_masks, len(self.mask_layers))
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
        mask_ids = th.arange(self.n_masks).to(self.fc1.weight.device)
        for l, l_w in zip(self.mask_layers, self.sparsity_layer_weights):
            # (n_masks, out_dim, in_dim)
            if self.weighted_masks:
                m = l._get_weighted_masks(l.weight, mask_ids)
            else:
                m = l._get_masks(l.weight, mask_ids)
            # every two masks distance
            loss += l_w * th.abs(m[None, :, :, :] - m[:, None, :, :]).sum()

        _, w_counts = self._get_linear_weight_sparsities()
        param_count = sum(w_counts)
        # we want to maximize the diversity
        # averaged over the number of agents and param_count to make it range from 0 to 1
        return -loss / (self.n_masks * (self.n_masks - 1)) / param_count

    def _reset_all_masks_weights(self, reset_ratio):
        if isinstance(reset_ratio, float):
            reset_ratio = [reset_ratio for _ in self.reset_layers]
        for l, r in zip(self.reset_layers, reset_ratio):
            l._reset(r)

    def _reset_mask(self, mask_id):
        for l in self.mask_layers:
            l._reset_mask(mask_id)