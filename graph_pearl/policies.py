from typing import Optional, List
import numpy as np
import torch
from torch import nn as nn
import torch_geometric as gtorch

from rlkit.core.util import Wrapper
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.core import PyTorchModule
from rlkit.torch.core import np_ify

from .graph_utils import Node
from .hetero_graph_modules import GraphModule

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Graph_TanhGaussianPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs, z)
    action, mean, log_std, _ = policy(obs, z, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, z, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            state_dim: int,
            latent_dim: int,
            action_dim: int,
            gnn_edge_types: int,
            gnn_node_count: int,
            gnn_layer_sizes: List[int],
            pre_gnn_layer_sizes: List[int] = [],
            post_gnn_layer_sizes: List[int] = [],
            split_layer_into_nodes: bool = False
    ):
        self.save_init_params(locals())
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.gnn_edge_types = gnn_edge_types
        self.gnn_node_count = gnn_node_count
        self.gnn_layer_sizes = gnn_layer_sizes
        self.pre_gnn_layer_sizes = pre_gnn_layer_sizes
        self.post_gnn_layer_sizes = post_gnn_layer_sizes
        self.split_layer_into_nodes = split_layer_into_nodes
        
        self.module = GraphModule(
            input_size=state_dim + latent_dim,
            output_size=2 * action_dim,
            gnn_edge_types=gnn_edge_types,
            gnn_node_count=gnn_node_count,
            gnn_layer_sizes=gnn_layer_sizes,
            pre_gnn_layer_sizes=pre_gnn_layer_sizes,
            post_gnn_layer_sizes=post_gnn_layer_sizes,
            split_layer_into_nodes=split_layer_into_nodes
        )

    def get_action(self, state: torch.Tensor, latent: Optional[torch.Tensor], graph_structure: torch.Tensor, deterministic: bool = False):
        actions = self.get_actions(state, latent, graph_structure, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, state: torch.Tensor, latent: Optional[torch.Tensor], graph_structure: torch.Tensor, deterministic: bool = False):
        outputs = self.forward(state, latent, graph_structure, deterministic=deterministic)[0]
        return np_ify(outputs)

    #@profile
    def forward(
            self,
            state: torch.Tensor,
            latent: Optional[torch.Tensor],
            graph_structure: torch.Tensor,
            reparameterize: bool = False,
            deterministic: bool = False,
            return_log_prob: bool = False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        network_out = self.module(state, latent, graph_structure=graph_structure)
        mean, log_std = torch.tensor_split(network_out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super().__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, *input):
        return self.stochastic_policy.get_action(*input,
                                                 deterministic=True)

    def get_actions(self, *input):
        return self.stochastic_policy.get_actions(*input,
                                                  deterministic=True)
