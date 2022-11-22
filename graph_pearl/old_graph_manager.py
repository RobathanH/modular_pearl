from typing import List, Dict, Optional
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as gtorch
import torch_geometric.nn as gnn

from rlkit.torch.core import PyTorchModule
from rlkit.torch.distributions import TanhNormal

class Node(Enum):
    STATE_IN = "state_in"
    ACTION_IN = "action_in"
    LATENT_IN = "latent_in"
    INNER = "inner"
    ACTION_MEAN_OUT = "action_mean_out"
    ACTION_LOG_STD_OUT = "action_log_std_out"
    Q1_OUT = "q1_out"
    Q2_OUT = "q2_out"
    V_OUT = "v_out"
    
    @staticmethod
    def all_types() -> List[str]:
        return [member.value for member in Node]
    
    @staticmethod
    def channel_dims(state_dim: int, action_dim: int, latent_dim: int, inner_dim: int) -> Dict[str, int]:
        return {
            Node.STATE_IN: state_dim,
            Node.ACTION_IN: action_dim,
            Node.LATENT_IN: latent_dim,
            Node.INNER: inner_dim,
            Node.ACTION_MEAN_OUT: action_dim,
            Node.ACTION_LOG_STD_OUT: action_dim,
            Node.Q1_OUT: 1,
            Node.Q2_OUT: 1,
            Node.V_OUT: 1
        }






LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class GraphModuleManager(PyTorchModule):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        inner_dim: int,
        inner_node_count: int,
        inner_edge_types: int,
        conv_iterations: int
    ):
        self.save_init_params(locals())
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.inner_dim = inner_dim
        self.inner_node_count = inner_node_count
        self.inner_edge_types = inner_edge_types
        self.conv_iterations = conv_iterations
        
        '''
        # List of edge names and corresponding channel counts
        edge_info = [
            ((Node.STATE_IN, "to", Node.INNER), state_dim, inner_dim),
            ((Node.ACTION_IN, "to", Node.INNER), action_dim, inner_dim),
            ((Node.LATENT_IN, "to", Node.INNER), latent_dim, inner_dim),
            ((Node.INNER, "to", Node.ACTION_MEAN_OUT), inner_dim, action_dim),
            ((Node.INNER, "to", Node.ACTION_LOG_STD_OUT), inner_dim, action_dim),
            ((Node.INNER, "to", Node.Q1_OUT), inner_dim, 1),
            ((Node.INNER, "to", Node.Q2_OUT), inner_dim, 1),
            ((Node.INNER, "to", Node.V_OUT), inner_dim, 1),
        ] + [
            ((Node.INNER, f"to_{i}", Node.INNER), inner_dim, inner_dim)
            for i in range(edge_types)
        ]
        
        self.conv = gnn.HeteroConv(
            {
                name: gnn.GATConv((source_channels, target_channels), target_channels)
                for name, source_channels, target_channels in edge_info
            },
            aggr="sum"
        )
        '''
        
        '''
        self.state_action_input_stage = gnn.HGTConv(
            in_channels={
                Node.STATE_IN: state_dim,
                Node.ACTION_IN: action_dim,
                Node.LATENT_IN: latent_dim,
                Node.INNER: inner_dim
            },
            out_channels=inner_dim,
            metadata=(
                [
                    Node.STATE_IN,
                    Node.ACTION_IN,
                    Node.LATENT_IN,
                    Node.INNER
                ],
                [
                    (Node.STATE_IN, "to", Node.INNER),
                    (Node.ACTION_IN, "to", Node.INNER),
                    (Node.LATENT_IN, "to", Node.INNER)
                ]
            )
        )
        self.state_input_stage = gnn.HGTConv(
            in_channels={
                Node.STATE_IN: state_dim,
                Node.LATENT_IN: latent_dim,
                Node.INNER: inner_dim
            },
            out_channels=inner_dim,
            metadata=(
                [
                    Node.STATE_IN,
                    Node.LATENT_IN,
                    Node.INNER
                ],
                [
                    (Node.STATE_IN, "to", Node.INNER),
                    (Node.LATENT_IN, "to", Node.INNER)
                ]
            )
        )
        
        self.inner_edges = [
            (Node.INNER, f"to_{i}", Node.INNER)
            for i in range(inner_edge_types)
        ]
        self.inner_stage = gnn.HGTConv(
            in_channels={
                Node.INNER: inner_dim
            },
            out_channels=inner_dim,
            metadata=(
                [
                    Node.INNER
                ],
                self.inner_edges
            )
        )
        
        self.policy_output_edges = [
            (Node.INNER, "to", Node.ACTION_MEAN_OUT),
            (Node.INNER, "to", Node.ACTION_LOG_STD_OUT)
        ]
        self.policy_output_stage = gnn.HGTConv(
            in_channels={
                Node.INNER: inner_dim,
                Node.ACTION_MEAN_OUT: action_dim,
                Node.ACTION_LOG_STD_OUT: action_dim
            },
            out_channels=action_dim,
            metadata=(
                [
                    Node.INNER,
                    Node.ACTION_MEAN_OUT,
                    Node.ACTION_LOG_STD_OUT
                ],
                self.policy_output_edges
            )
        )
        
        self.q_output_stage = gnn.HGTConv(
            in_channels={
                Node.INNER: inner_dim,
                Node.Q1_OUT: 1,
                Node.Q2_OUT: 1
            },
            out_channels=1,
            metadata=(
                [
                    Node.INNER,
                    Node.Q1_OUT,
                    Node.Q2_OUT
                ],
                [
                    (Node.INNER, "to", Node.Q1_OUT),
                    (Node.INNER, "to", Node.Q2_OUT)
                ]
            )
        )
        self.v_output_stage = gnn.HGTConv(
            in_channels={
                Node.INNER: inner_dim,
                Node.V_OUT: 1
            },
            out_channels=1,
            metadata=(
                [
                    Node.INNER,
                    Node.V_OUT
                ],
                [
                    (Node.INNER, "to", Node.V_OUT)
                ]
            )
        )
        '''
        
        self.conv = gnn.HGTConv(
            in_channels={
                Node.STATE_IN: state_dim,
                Node.ACTION_IN: action_dim,
                Node.LATENT_IN: latent_dim,
                Node.INNER: inner_dim,
                Node.ACTION_MEAN_OUT: inner_dim,
                Node.ACTION_LOG_STD_OUT: inner_dim,
                Node.Q1_OUT: inner_dim,
                Node.Q2_OUT: inner_dim,
                Node.V_OUT: inner_dim
            },
            out_channels=inner_dim,
            metadata=(
                [
                    Node.STATE_IN,
                    Node.ACTION_IN,
                    Node.LATENT_IN,
                    Node.INNER,
                    Node.ACTION_MEAN_OUT,
                    Node.ACTION_LOG_STD_OUT,
                    Node.Q1_OUT,
                    Node.Q2_OUT,
                    Node.V_OUT
                ],
                [
                    (Node.STATE_IN, "to", Node.INNER),
                    (Node.ACTION_IN, "to", Node.INNER),
                    (Node.LATENT_IN, "to", Node.INNER),
                    (Node.INNER, "to", Node.ACTION_MEAN_OUT),
                    (Node.INNER, "to", Node.ACTION_LOG_STD_OUT),
                    (Node.INNER, "to", Node.Q1_OUT),
                    (Node.INNER, "to", Node.Q2_OUT),
                    (Node.INNER, "to", Node.V_OUT),
                ] + [
                    (Node.INNER, f"to_{i}", Node.INNER)
                    for i in range(inner_edge_types)
                ]
            )
        )
        self.to_action_mean = nn.Linear(inner_dim, action_dim)
        self.to_action_log_std = nn.Linear(inner_dim, action_dim)
        self.to_q1 = nn.Linear(inner_dim, 1)
        self.to_q2 = nn.Linear(inner_dim, 1)
        self.to_v = nn.Linear(inner_dim, 1)
        
        
    
    def policy_forward(
        self,
        state: torch.Tensor,
        latent_graph: gtorch.data.HeteroData,
        latent_state: torch.Tensor,
        reparameterize: bool = False,
        deterministic: bool = False,
        return_log_prob: bool = False
    ) -> torch.Tensor:
        
        # Remove irrelevant node types
        latent_graph = latent_graph.node_type_subgraph([
            Node.STATE_IN,
            Node.LATENT_IN,
            Node.INNER,
            Node.ACTION_MEAN_OUT,
            Node.ACTION_LOG_STD_OUT
        ])
        
        # Make sure sizes are consistent
        assert latent_graph[Node.STATE_IN].x.shape == state.shape
        assert latent_graph[Node.LATENT_IN].x.shape == latent_state.shape
        assert latent_graph[Node.INNER].x.shape[0] == state.shape[0]
        assert latent_graph[Node.ACTION_MEAN_OUT].x.shape[:2] == state.shape[:2]
        assert latent_graph[Node.ACTION_LOG_STD_OUT].x.shape[:2] == state.shape[:2]
        
        # Fill in relevant input values into nodes
        latent_graph[Node.STATE_IN].x = state
        latent_graph[Node.LATENT_IN].x = latent_state
        
        # Make sure all other node types are initialized to zero
        latent_graph[Node.INNER].x.zero_()
        latent_graph[Node.ACTION_MEAN_OUT].x.zero_()
        latent_graph[Node.ACTION_LOG_STD_OUT].x.zero_()
        
        # Run GNN
        x_dict = latent_graph.x_dict
        edge_index_dict = latent_graph.edge_index_dict
        for i in range(self.conv_iterations):
            x_dict = self.conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(val) for key, val in x_dict.items()}
            
        mean = self.to_action_mean(x_dict[Node.ACTION_MEAN_OUT])
        log_std = self.to_action_log_std(x_dict[Node.ACTION_LOG_STD_OUT])
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
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )
        
        
        
    def q1_forward(
        self,
        
    )