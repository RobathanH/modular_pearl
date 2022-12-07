from typing import List, Dict, Optional
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as gtorch
import torch_geometric.nn as gnn

from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu


'''
Reimplement torch_geometric.nn.GCNConv to allow operations on batched node features
'''

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, self.flow)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        x_feature_dims = x_j.dim() - 1
        return x_j if edge_weight is None else edge_weight.view(-1, *(1,) * x_feature_dims) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


'''
Neural Network including graph convolution stages over multiple edge types
'''

class GraphModule(PyTorchModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 gnn_edge_types: int,
                 gnn_layer_sizes: List[int],
                 pre_gnn_layer_sizes: List[int] = [],
                 post_gnn_layer_sizes: List[int] = []):
        self.save_init_params(locals())
        super().__init__()
        
        self.gnn_edge_types = gnn_edge_types
        
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
        
        pre_gnn_size = input_size if len(pre_gnn_layer_sizes) == 0 else pre_gnn_layer_sizes[-1]
        self.gnn_layers = nn.ModuleList([
            gnn.HeteroConv({
                ("node", f"to_{i}", "node"): GCNConv(
                    in_size, out_size,
                    node_dim=0, add_self_loops=False
                )
                for i in range(gnn_edge_types)
            })
            for in_size, out_size in zip(
                [pre_gnn_size] + gnn_layer_sizes[:-1],
                gnn_layer_sizes
            )
        ])
        
        self.gnn_pool = gnn.aggr.MeanAggregation()
        
        fc_out_layer_list = []
        for in_size, out_size in zip(
            [gnn_layer_sizes[-1]] + post_gnn_layer_sizes,
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
        num_tasks, node_count, _ = graph_structure.size()
        
        input_features = [x for x in input_features if x is not None]
        input_features = torch.cat(input_features, dim=-1)
        
        pre_gnn = self.fc_in(input_features)
        
        # One for each task (separate graph), copy/repeat for each node within a task
        node_features = torch.repeat_interleave(pre_gnn, node_count, dim=0)
        node_task_indices = torch.repeat_interleave(ptu.arange(num_tasks), node_count)
        
        # Differs between tasks, one for each node-pair in each task
        # Referenced node indices must be incremented so they don't overlap between tasks
        orig_edge_index = (graph_structure < self.gnn_edge_types).nonzero().transpose(0, 1).long()
        edge_index = orig_edge_index[1:, :] + orig_edge_index[0, :].unsqueeze(0) * node_count # Increment node indices based on which task they're from
        edge_type = graph_structure[orig_edge_index[0], orig_edge_index[1], orig_edge_index[2]]
        
        x_dict = {"node": node_features}
        edge_index_dict = {
            ("node", f"to_{i}", "node"): edge_index[:, (edge_type == i).nonzero()[:, 0].long()]
            for i in range(self.gnn_edge_types)
        }
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            if i != 0:
                x_dict = {key: F.relu(val) for key, val in x_dict.items()}
            x_dict = gnn_layer(x_dict, edge_index_dict)
            
        post_gnn = self.gnn_pool(x_dict["node"], index=node_task_indices, dim=0)
        
        output = self.fc_out(post_gnn)
        return output