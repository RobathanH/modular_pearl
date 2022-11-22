from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch import nn as nn
import torch_geometric as gtorch

import rlkit.torch.pytorch_util as ptu

'''
Enum for all possible node types
'''
class Node:
    STATE_IN = "state_in"
    ACTION_IN = "action_in"
    LATENT_IN = "latent_in"
    INNER = "inner"



def get_inner_edge_keys(inner_edge_types: int) -> List[Tuple[int, int, int]]:
    return [(Node.INNER, f"to_{i}", Node.INNER) for i in range(inner_edge_types)]

def construct_graph(inner_node_count: int, inner_edge_types: int, inner_dim: int,
                    structure: torch.Tensor, state: torch.Tensor, latent: torch.Tensor,
                    action: Optional[torch.Tensor] = None) -> gtorch.data.HeteroData:
    """Create a torch_geometric graph structure based on the inner node edge structure given,
    and add input nodes of relevant types, filling in with the data given.

    Args:
        inner_node_count (int): Number of inner nodes in graph
        inner_edge_types (int): Number of potential types of inner->inner edges
        inner_dim (int): Dimensionality of inner node features
        structure (torch.Tensor): Shape = (num_tasks, inner_node_count, inner_node_count), Value = int {0, ..., inner_edge_types}
        state (torch.Tensor): Shape = (num_tasks, batch_size, state_dim)
        latent (torch.Tensor): Shape = (num_tasks, batch_size, latent_dim)
        action (Optional[torch.Tensor], optional): Shape = (num_tasks, batch_size, action_dim). Defaults to None.

    Returns:
        gtorch.data.HeteroData: Output data with inner nodes initialized to zeros, and input nodes initialized with given values.
    """
    
    num_tasks = state.size(0)
    data_batch_size = state.size(1)
    
    
    graph_list = []
    for t in range(num_tasks):
        graph = gtorch.data.HeteroData()
        graph[Node.INNER].x = ptu.zeros(inner_node_count, data_batch_size, inner_dim)
        for i, inner_edge_key in enumerate(get_inner_edge_keys(inner_edge_types)):
            graph[inner_edge_key].edge_index = (structure[t] == i).nonzero().transpose(0, 1).long()
        
        graph[Node.STATE_IN].x = state[t].unsqueeze(0)
        graph[Node.STATE_IN, "to", Node.INNER].edge_index = torch.vstack([ptu.zeros(inner_node_count), ptu.arange(inner_node_count)]).long()
        
        graph[Node.LATENT_IN].x = latent[t].unsqueeze(0)
        graph[Node.LATENT_IN, "to", Node.INNER].edge_index = torch.vstack([ptu.zeros(inner_node_count), ptu.arange(inner_node_count)]).long()
        
        if action is not None:
            graph[Node.ACTION_IN].x = action[t].unsqueeze(0)
            graph[Node.ACTION_IN, "to", Node.INNER].edge_index = torch.vstack([ptu.zeros(inner_node_count), ptu.arange(inner_node_count)]).long()
    
        graph_list.append(graph)
    
    # Graph batch construction treats nodes as feature dim 0, later operators must specify node_dim=0
    output = gtorch.data.Batch.from_data_list(graph_list)
    
    return output