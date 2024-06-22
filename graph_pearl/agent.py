from typing import Union
from tqdm import trange, tqdm
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import MlpEncoder, RecurrentEncoder

from .policies import Graph_TanhGaussianPolicy
from .graph_utils import construct_graph, get_inner_edge_keys


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Graph_PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 context_vector_encoder: Union[MlpEncoder, RecurrentEncoder],
                 context_graph_encoder: Union[MlpEncoder, RecurrentEncoder],
                 policy: Graph_TanhGaussianPolicy,
                 qf1,
                 qf2,
                 vf,
                 **kwargs
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Treat latent_dim = 0 as a flag for special mode where latent is set to None for all modules
        # Effectively ignores latent without having to disable all infrastructure which assumes latent exists
        if latent_dim == 0:
            latent_dim = 1
            self.latent_dim = 1
            self.disable_vector_latent = True
        else:
            self.disable_vector_latent = False
        
        self.gnn_node_count = kwargs['gnn_node_count']
        self.gnn_edge_types = kwargs['gnn_edge_types']
        self.sim_anneal_train_proposals = kwargs['sim_anneal_train_proposals']
        self.sim_anneal_eval_proposals = kwargs['sim_anneal_eval_proposals']
        self.sim_anneal_temp = self.sim_anneal_init_temp = kwargs['sim_anneal_init_temp']
        self.sim_anneal_eval_temp_decay = kwargs.get('sim_anneal_eval_temp_decay', 0.95)

        self.context_vector_encoder = context_vector_encoder # (context -> z_means, z_vars)
        self.context_graph_encoder = context_graph_encoder # (context -> inner_edge_probs (innner_node_count, gnn_node_count, gnn_edge_types))
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.reward_scale = kwargs['reward_scale']
        self.discount = kwargs['discount']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))
        self.register_buffer('graph_structure', torch.zeros(1, self.gnn_node_count, self.gnn_node_count))
        self.register_buffer('graph_structure_probs', torch.zeros(1, self.gnn_node_count, self.gnn_node_count, self.gnn_edge_types))

        # Maintain statistics about structure annealing acceptance rate
        self.reset_annealing_statistics()

        self.clear_z()

    #@profile
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.graph_structure_probs = ptu.ones(num_tasks, self.gnn_node_count, self.gnn_node_count, self.gnn_edge_types) / (self.gnn_edge_types)
        self.latest_used_context = None # Save the context tuples which correspond to the current graph structure probs, allows resampling and reannealing graph
        
        # sample a new z from the prior
        self.sample_z()
        self.sample_graph()
        
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_vector_encoder.reset(num_tasks)
        self.context_graph_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_vector_encoder.hidden = self.context_vector_encoder.hidden.detach()

    #@profile
    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        d = ptu.from_numpy(np.array([[d]]))

        data = [o, a, r, no, d]
        if self.context is None:
            self.context = data
        else:
            self.context = [torch.cat([context_component, new_component], dim=1) for context_component, new_component in zip(self.context, data)]

    #@profile
    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        '''
        vector_prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        vector_posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        vector_kl_divs = [torch.distributions.kl.kl_divergence(post, vector_prior) for post in vector_posteriors]
        vector_kl_div_sum = torch.sum(torch.stack(vector_kl_divs))
        '''
        # Closed-form KL-divergence against unit prior
        vector_kl_div_per_dim = 0.5 * (-torch.log(self.z_vars) - 1 + torch.square(self.z_means) + self.z_vars).sum(dim=-1)
        vector_kl_div_sum = vector_kl_div_per_dim.sum()
        
        ''' compute categorical KL div for graph latent encoder '''
        '''
        graph_prior = torch.distributions.Categorical(ptu.ones(self.gnn_edge_types) / self.gnn_edge_types)
        graph_posteriors = [torch.distributions.Categorical(edge_probs) for edge_probs in torch.unbind(self.graph_structure_probs.view(-1, self.gnn_edge_types))]
        graph_kl_divs = [torch.distributions.kl.kl_divergence(post, graph_prior) for post in graph_posteriors]
        graph_kl_div_sum = torch.sum(torch.stack(graph_kl_divs))
        '''
        graph_kl_divs = (self.graph_structure_probs * (self.gnn_edge_types * self.graph_structure_probs + 1e-9).log()).sum(dim=-1)
        graph_kl_div_sum = graph_kl_divs.sum()
        
        return vector_kl_div_sum, graph_kl_div_sum

    #@profile
    def infer_posterior(self, context, sample_graph=True, anneal_graph=True, train_annealing=False):
        # Package context components for probabilistic encoders, which takes flattened vec inputs
        if self.use_next_obs_in_context:
            packed_context = torch.cat(context[:-1], dim=2)
        else:
            packed_context = torch.cat(context[:-2], dim=2)
        
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_vector_encoder(packed_context)
        params = params.view(packed_context.size(0), -1, self.context_vector_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()
            
        '''
        Infer graph posterior, then sample from it (must sample graph after sampling z), then anneal
        '''
        '''
        # Outputs independent dirichlet parameters for each context tuple and for node pair
        # Conjugate categorical distribution is based on sum of all those parameters
        graph_edge_params = F.relu(self.context_graph_encoder(packed_context)) + 1
        graph_edge_params = graph_edge_params.view(packed_context.size(0), packed_context.size(1), self.gnn_node_count, self.gnn_node_count, self.gnn_edge_types + 1)
        graph_edge_params = graph_edge_params.sum(dim=1)
        self.graph_structure_probs = graph_edge_params / graph_edge_params.sum(dim=-1, keepdim=True)
        '''
        graph_edge_prob_logits = self.context_graph_encoder(packed_context).view(*packed_context.shape[:2], self.gnn_node_count, self.gnn_node_count, self.gnn_edge_types)
        graph_edge_prob_logits = graph_edge_prob_logits.sum(dim=1)
        self.graph_structure_probs = F.softmax(graph_edge_prob_logits, dim=-1)
        self.latest_used_context = context
        
        if sample_graph:
            self.sample_graph()
        if anneal_graph:
            self.anneal_graph_structure(train_annealing)

    #@profile
    def sample_z(self):
        if self.use_ib:
            '''
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
            '''
            flat_posterior = torch.distributions.Normal(self.z_means.flatten(), torch.sqrt(self.z_vars.flatten()))
            flat_z = flat_posterior.rsample()
            self.z = flat_z.reshape(self.z_means.shape)
        else:
            self.z = self.z_means
           
    @torch.no_grad()
    #@profile
    def sample_graph(self):
        # Use Structure Proposal probabilistically
        if self.use_ib:
            self.graph_structure = torch.multinomial(
                self.graph_structure_probs.flatten(0, -2),
                num_samples=1
            )[:, 0].reshape(self.graph_structure_probs.shape[:-1])
        
        # Use structure proposal deterministically
        else:
            self.graph_structure = self.graph_structure_probs.detach().argmax(dim=-1)
       
    @torch.no_grad()
    #@profile
    def graph_structure_loss(self, context, proposed_graph_structure):
        context_obs, context_act, context_r, context_next_obs, context_terms = context
        
        '''
        state_action_input = construct_graph(self.gnn_node_count, self.gnn_edge_types, self.inner_dim,
                                             proposed_graph_structure, context_obs, self.z, action=context_act)
        next_state_input = construct_graph(self.gnn_node_count, self.gnn_edge_types, self.inner_dim,
                                           proposed_graph_structure, context_next_obs, self.z, action=None)
        #print(state_action_input)
        #print(state_action_input.x_dict)
        #print(state_action_input.edge_index_dict)
        q1_pred = self.qf1(state_action_input)
        q2_pred = self.qf2(state_action_input)
        v_pred = self.vf(next_state_input)
        #print("q1", q1_pred)
        #print("q2", q2_pred)
        #print("v", v_pred)
        '''
        latent = self.z.unsqueeze(1).expand(-1, context_obs.size(1), -1)
        if self.disable_vector_latent:
            latent = None
        q1_pred = self.qf1(context_obs, context_act, latent, graph_structure=proposed_graph_structure)
        q2_pred = self.qf2(context_obs, context_act, latent, graph_structure=proposed_graph_structure)
        v_pred = self.vf(context_next_obs, latent, graph_structure=proposed_graph_structure)
        
        q_target = context_r * self.reward_scale + \
            (1. - context_terms) * self.discount * v_pred
        q_loss_per_task = torch.mean((q1_pred - q_target) ** 2, dim=[1,2]) + torch.mean((q2_pred - q_target) ** 2, dim=[1,2])
        return q_loss_per_task
               
    @torch.no_grad() 
    #@profile
    def anneal_graph_structure(self, train_annealing = False):
        # Use the context used to compute self.graph_structure_probs for annealing
        context = self.latest_used_context
        
        # No annealing possible without context, we just sample from prior
        if context is None:
            return
        
        num_tasks = self.z.size(0)
        
        curr_structure_loss = self.graph_structure_loss(context, self.graph_structure)
        all_structure_losses = [curr_structure_loss]
        all_proposed_structure_losses = []
        
        if train_annealing:
            proposal_count = self.sim_anneal_train_proposals
            temp = self.sim_anneal_temp
        else:
            proposal_count = self.sim_anneal_eval_proposals
            temp = self.sim_anneal_init_temp
            
        for step in trange(proposal_count, desc="annealing", leave=False, delay=0.5):
            node_i = ptu.randint(0, self.gnn_node_count, size=(num_tasks,))
            node_j = ptu.randint(0, self.gnn_node_count, size=(num_tasks,))
            
            # Get probabilities for each chosen node-pair from graph structure encoder,
            proposal_edge_type_probs = self.graph_structure_probs[ptu.arange(num_tasks), node_i, node_j].detach().clone()
            proposal_edge_type_probs += 1e-5 # to avoid any zero-weight errors in multinomial if all prob is at current edge

            # then zero-out probability for current edge type (so proposal is definitely different from current)
            curr_edge_types = self.graph_structure[ptu.arange(num_tasks), node_i, node_j]
            proposal_edge_type_probs[ptu.arange(num_tasks), curr_edge_types] = 0

            # Sample proposal edge types from graph structure encoder edge-type distribution
            proposal_edge_types = torch.multinomial(proposal_edge_type_probs, num_samples=1)[:, 0]
            
            # Apply proposed changes for each task subgraph
            proposed_structure = self.graph_structure.clone()
            proposed_structure[ptu.arange(num_tasks), node_i, node_j] = proposal_edge_types
            proposed_structure_loss = self.graph_structure_loss(context, proposed_structure)
            #print(node_i.tolist(), node_j.tolist(), edge_type.tolist())
            #print(f"orig: {self.graph_structure}")
            #print(f"proposed: {proposed_structure}")
            
            # Accept or reject change for each task
            prob_accept = ((curr_structure_loss - proposed_structure_loss) / temp).exp()
            accept = (ptu.rand(size=prob_accept.shape) <= prob_accept)
            accepted_task_inds = accept.nonzero()
            
            if train_annealing:
                worse = (proposed_structure_loss > curr_structure_loss)
                self.structure_worse_proposals_count += worse.sum().item()
                self.structure_worse_proposals_accepted += (worse & accept).sum().item()
            
            #print(prob_accept.tolist())
            #print(accept.tolist())
            #print(accepted_task_inds.tolist())
            #print(node_i[accepted_task_inds].tolist(), node_j[accepted_task_inds].tolist(), edge_type[accepted_task_inds].tolist())
            
            if len(accepted_task_inds):
                self.graph_structure[accepted_task_inds, node_i[accepted_task_inds], node_j[accepted_task_inds]] = proposal_edge_types[accepted_task_inds]
                curr_structure_loss = torch.where(accept, proposed_structure_loss, curr_structure_loss)
                
            all_structure_losses.append(curr_structure_loss)
            all_proposed_structure_losses.append(proposed_structure_loss)
            
            if not train_annealing:
                temp *= self.sim_anneal_eval_temp_decay
                
        self.structure_loss_std += torch.std(torch.vstack(all_proposed_structure_losses), dim=0).tolist()
        #print(f"Final Structure Loss: {curr_structure_loss.tolist()}")
        #print(f"Accept Rate: {total_accepted / (self.sim_anneal_proposals * num_tasks)}")
        #print(f"Structure Loss Std: {torch.std(torch.vstack(all_structure_losses), dim=0)}")
        #print(f"Proposed Structure Loss Std: {torch.std(torch.vstack(all_proposed_structure_losses), dim=0)}")

    @torch.no_grad()
    #@profile
    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        state = ptu.from_numpy(obs[None, None])
        latent = self.z.unsqueeze(1).expand(-1, state.size(1), -1)
        if self.disable_vector_latent:
            latent = None
        
        '''
        filled_graph = construct_graph(self.gnn_node_count, self.gnn_edge_types, self.inner_dim,
                                    self.graph_structure, state, latent, action=None)
        
        action, info = self.policy.get_action(filled_graph, deterministic=deterministic)
        '''
        action, info = self.policy.get_action(state, latent, graph_structure=self.graph_structure)
        return action[0], info

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    # Assumes latent and graph structure have been inferred from context and sampled already
    #@profile
    def forward(self, obs, context, sample_graph=True, anneal_graph=True):
        self.infer_posterior(context, sample_graph=sample_graph, anneal_graph=anneal_graph, train_annealing=True)
        
        t, b, _ = obs.size()
        state = obs
        latent = self.z.view(t, 1, -1).expand(-1, b, -1).detach()
        if self.disable_vector_latent:
            latent = None
        
        '''
        filled_graph = construct_graph(self.gnn_node_count, self.gnn_edge_types, self.inner_dim,
                                       graph_structure, state, latent, action=None)

        policy_outputs = self.policy(filled_graph, reparameterize=True, return_log_prob=True)
        '''
        policy_outputs = self.policy(state, latent, graph_structure=self.graph_structure, reparameterize=True, return_log_prob=True)
        
        return policy_outputs

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

        eval_statistics['SA Temp'] = self.sim_anneal_temp
        if self.structure_worse_proposals_count != 0:
            eval_statistics['SA Accept Rate'] = self.structure_worse_proposals_accepted / self.structure_worse_proposals_count
        else:
            eval_statistics['SA Accept Rate'] = np.nan
        eval_statistics['SA Structure Loss Std'] = sum(self.structure_loss_std) / len(self.structure_loss_std)

    @property
    def networks(self):
        return [self.context_vector_encoder, self.context_graph_encoder, self.policy, self.qf1, self.qf2, self.vf]

    # Manage running statistics about graph structure annealing
    def reset_annealing_statistics(self):
        self.structure_worse_proposals_count = 0
        self.structure_worse_proposals_accepted = 0
        self.structure_loss_std = [] # std of structure loss for proposals for particular tasks

    # Update temperature with a small fixed multiplicative factor to move towards goal accept rate
    def update_temp(self, goal_acc_rate: float) -> None:
        if self.structure_worse_proposals_count == 0:
            return

        temp_change_factor = 1.1
        current_accept_rate = self.structure_worse_proposals_accepted / self.structure_worse_proposals_count
        if current_accept_rate < goal_acc_rate:
            self.sim_anneal_temp *= temp_change_factor
        else:
            self.sim_anneal_temp /= temp_change_factor
