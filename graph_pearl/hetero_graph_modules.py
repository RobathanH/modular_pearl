from typing import List, Dict, Optional
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as gtorch
import torch_geometric.nn as gnn
from torch_scatter import scatter_add

from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu


'''
Implement optimized multi-edge-type version of gnn.conv.GCNConv
'''
#@profile
def gcn_norm_edge_weights(edge_index: torch.Tensor, num_nodes: int,
                          flow="source_to_target", dtype=None):

    assert flow in ["source_to_target", "target_to_source"]

    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    


class ManualHeteroGCNConv(PyTorchModule):
    def __init__(self, in_channels: int, out_channels: int, edge_type_count: int,
                 normalize: bool = True, bias: bool = True, node_dim=0):
        self.save_init_params(locals())
        super().__init__()
        
        assert node_dim == 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_type_count = edge_type_count
        self.normalize = normalize
        
        self.lin = nn.Linear(in_channels, out_channels * edge_type_count, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)
        
    #@profile
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        device = x.device
        
        if self.normalize:
            edge_weights = gcn_norm_edge_weights(edge_index, num_nodes)
        else:
            edge_weights = torch.ones(edge_index.size(1), device=edge_index.device)
            
        # Compute linear transform for each potential edge type simultaneously
        # Message stage of self.propagate() will select the correct edge types
        source_node_messages_all_types = self.lin(x).reshape(*x.shape[:-1], self.out_channels, self.edge_type_count)
        
        # Collect correct messages for each edge type
        edge_to_source_node_mask = F.one_hot(edge_index[0], num_classes=num_nodes).float().to(device)
        edge_messages_all_types = torch.tensordot(edge_to_source_node_mask, source_node_messages_all_types, dims=1)
        
        # Reduce to correct edge type for each edge
        edge_to_edge_type_mask = F.one_hot(edge_types, num_classes=self.edge_type_count).to(device).view(num_edges, *(1,) * (edge_messages_all_types.dim() - 2), self.edge_type_count)
        edge_messages = (edge_messages_all_types * edge_to_edge_type_mask).sum(dim=-1)
        
        # Weigh each edge message
        edge_weights = edge_weights.view(num_edges, *(1,) * (edge_messages.dim() - 1))
        weighted_edge_messages = edge_messages * edge_weights
        
        # Aggregate edge messages to target nodes
        target_node_to_edge_mask = F.one_hot(edge_index[1], num_classes=num_nodes).float().to(device).T
        target_node_outputs = torch.tensordot(target_node_to_edge_mask, weighted_edge_messages, dims=1)
        
        out = target_node_outputs
        
        if self.bias is not None:
            out += self.bias
            
        return out
        
        

        
'''
Neural Network including graph convolution stages over multiple edge types
'''

class GraphModule(PyTorchModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 gnn_edge_types: int,
                 gnn_node_count: int,
                 gnn_layer_sizes: List[int],
                 pre_gnn_layer_sizes: List[int] = [],
                 post_gnn_layer_sizes: List[int] = [],
                 split_layer_into_nodes: bool = False):
        self.save_init_params(locals())
        super().__init__()
        
        self.gnn_edge_types = gnn_edge_types
        self.gnn_node_count = gnn_node_count
        self.split_layer_into_nodes = split_layer_into_nodes
        
        if len(pre_gnn_layer_sizes) == 0:
            self.fc_in = nn.Identity()
        else:
            fc_in_layer_list = []
            for in_size, out_size in zip(
                [input_size] + pre_gnn_layer_sizes[:-1],
                pre_gnn_layer_sizes
            ):
                if len(fc_in_layer_list):
                    fc_in_layer_list.append(nn.ReLU())
                fc_in_layer_list.append(nn.Linear(in_size, out_size))
            self.fc_in = nn.Sequential(*fc_in_layer_list)
        
        gnn_input_size = input_size if len(pre_gnn_layer_sizes) == 0 else pre_gnn_layer_sizes[-1]
        
        if split_layer_into_nodes:
            assert gnn_input_size % gnn_node_count == 0, "Pre-gnn layer size must be able to split evenly into node features"
            gnn_input_size = int(gnn_input_size / gnn_node_count)
        
        self.gnn_layers = nn.ModuleList([
            ManualHeteroGCNConv(in_size, out_size, gnn_edge_types, normalize=False, node_dim=0)
            for in_size, out_size in zip(
                [gnn_input_size] + gnn_layer_sizes[:-1],
                gnn_layer_sizes
            )
        ])
        
        post_gnn_input_size = gnn_layer_sizes[-1]
        if split_layer_into_nodes:
            post_gnn_input_size *= gnn_node_count
        
        fc_out_layer_list = []
        for in_size, out_size in zip(
            [post_gnn_input_size] + post_gnn_layer_sizes,
            post_gnn_layer_sizes + [output_size]
        ):
            if len(fc_out_layer_list):
                fc_out_layer_list.append(nn.ReLU())
            fc_out_layer_list.append(nn.Linear(in_size, out_size))
        self.fc_out = nn.Sequential(*fc_out_layer_list)
        
    #@profile
    def forward(self, *input_features: torch.Tensor, graph_structure: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input_features (torch.Tensor): Shape = (num_tasks, batch_size, input_dim)
            graph_structure (torch.Tensor): Shape = (num_tasks, node_count, node_count) in {0, ..., gnn_edge_types}

        Returns:
            torch.Tensor: Shape = (num_tasks, batch_size, output_dim)
        """
        input_features = [x for x in input_features if x is not None]
        input_features = torch.cat(input_features, dim=-1)
        
        num_tasks, batch_size, _ = input_features.size()
        
        pre_gnn = self.fc_in(input_features)
        
        if self.split_layer_into_nodes:
            node_features = pre_gnn.reshape(num_tasks, batch_size, self.gnn_node_count, -1).transpose(1, 2).reshape(num_tasks * self.gnn_node_count, batch_size, -1)
        else:
            # One for each task (separate graph), copy/repeat for each node within a task
            node_features = pre_gnn.unsqueeze(1).expand(-1, self.gnn_node_count, *(-1,) * (pre_gnn.dim() - 1)).flatten(0, 1)
        
        # Differs between tasks, one for each node-pair in each task
        # Referenced node indices must be incremented so they don't overlap between tasks
        orig_edge_index = (graph_structure < self.gnn_edge_types).nonzero().transpose(0, 1).long()
        edge_index = orig_edge_index[1:, :] + orig_edge_index[0, :].unsqueeze(0) * self.gnn_node_count # Increment node indices based on which task they're from
        edge_types = graph_structure[orig_edge_index[0], orig_edge_index[1], orig_edge_index[2]]
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            if i != 0:
                node_features = F.relu(node_features)
            node_features = gnn_layer(node_features, edge_index, edge_types)
            
        if self.split_layer_into_nodes:
            # Concatenate nodes for each task back into full feature layers
            post_gnn = node_features.reshape(num_tasks, self.gnn_node_count, batch_size, -1).transpose(1, 2).reshape(num_tasks, batch_size, -1)
        else:
            # Manually do mean pooling for each task
            node_task_indices = ptu.arange(num_tasks * self.gnn_node_count) // self.gnn_node_count
            task_to_node_mask = F.one_hot(node_task_indices).float().to(ptu.device).T
            task_to_node_mask /= task_to_node_mask.sum(dim=1, keepdim=True)
            post_gnn = torch.tensordot(task_to_node_mask, node_features, dims=1)
        post_gnn = F.relu(post_gnn)
        
        output = self.fc_out(post_gnn)
        return output