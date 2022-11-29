from collections import OrderedDict
import numpy as np
import gtimer as gt

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import torch_geometric as gtorch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

from .graph_utils import construct_graph


class Graph_PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            agent,

            inner_node_count=3,
            inner_edge_types=2,
            inner_dim=200,
            graph_conv_iterations=4,
            context_graph_lr=1e-3,
            sim_anneal_proposals=10,
            bouncegrad_iterations=10,
            graph_kl_lambda=0.1,
            sim_anneal_init_temp=1E-4,
            sim_anneal_init_goal_acc_rate=0.3,
            sim_anneal_final_goal_acc_rate=2E-3,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=agent,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        
        # Check for special mode where vector latents are ignored, and only
        # graph structure latent is used
        self.disable_vector_latent = (latent_dim == 0)
        
        self.inner_node_count = inner_node_count
        self.inner_edge_types = inner_edge_types
        self.inner_dim = inner_dim
        self.bouncegrad_iterations = bouncegrad_iterations
        self.graph_kl_lambda = graph_kl_lambda
        self.sim_anneal_init_goal_acc_rate = sim_anneal_init_goal_acc_rate
        self.sim_anneal_final_goal_acc_rate = sim_anneal_final_goal_acc_rate
        self.sim_anneal_goal_acc_rate_decay_factor = np.exp(np.log(sim_anneal_final_goal_acc_rate / sim_anneal_init_goal_acc_rate) / (kwargs['num_iterations'] * kwargs['num_train_steps_per_itr']))

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = agent.qf1, agent.qf2, agent.vf
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_vector_optimizer = optimizer_class(
            self.agent.context_vector_encoder.parameters(),
            lr=context_lr,
        )
        self.context_graph_optimizer = optimizer_class(
            self.agent.context_graph_encoder.parameters(),
            lr=context_graph_lr
        )
        
        # Save running goal sim-annealing accept rate, which decays over time
        self.sim_anneal_goal_acc_rate = sim_anneal_init_goal_acc_rate

    ###### Torch stuff #####
    @property
    def networks(self):
        return [self.agent] + self.agent.networks + [self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        # Include individual context parts separately for simulated annealing step
        obs, act, rewards, next_obs, terms = context
            
        return obs, act, rewards, next_obs, terms

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        
        # Update goal sim-anneal accept rate
        self.sim_anneal_goal_acc_rate *= self.sim_anneal_goal_acc_rate_decay_factor
        
        # Update agent temperature towards goal acc rate (if previous SA statistics exist)
        self.agent.update_temp(self.sim_anneal_goal_acc_rate)
        
        # Reset SA statistics
        self.agent.reset_annealing_statistics()

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = tuple(
                context_component[:, i * mb_size: i * mb_size + mb_size, :]
                for context_component in context_batch
            )
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, latent, graph_structure):
        '''
        state_action_input = construct_graph(self.inner_node_count, self.inner_edge_types, self.inner_dim,
                                             graph_structure, obs, latent.detach(), action=actions)
        q1 = self.qf1(state_action_input)
        q2 = self.qf2(state_action_input)
        '''
        q1 = self.qf1(obs, actions, latent, graph_structure=graph_structure)
        q2 = self.qf2(obs, actions, latent, graph_structure=graph_structure)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):
        num_tasks = len(indices)

        # data is (task, batch, feat) (and is not flattened later)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        
        for bouncegrad_it in range(self.bouncegrad_iterations):
            
            if bouncegrad_it != 0:
                self.agent.anneal_graph_structure(context)
            
            # run inference in networks
            # Don't update graph structure in later iterations, instead we'll manually anneal
            policy_outputs = self.agent(obs, context, update_graph_structure=(bouncegrad_it == 0))
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            graph_structure = self.agent.graph_structure
            
            if self.disable_vector_latent:
                latent = detached_latent = None
            else:
                latent = self.agent.z.unsqueeze(1).expand(-1, obs.size(1), -1)
                detached_latent = latent.detach()

            # Q and V networks
            # encoder will only get gradients from Q nets (through state_action_input)
            '''
            state_action_input = construct_graph(self.inner_node_count, self.inner_edge_types, self.inner_dim,
                                                graph_structure, obs, latent, action=actions)
            q1_pred = self.qf1(state_action_input)
            q2_pred = self.qf2(state_action_input)
            state_input = construct_graph(self.inner_node_count, self.inner_edge_types, self.inner_dim,
                                        graph_structure, obs, latent.detach())
            v_pred = self.vf(state_input)
            # get targets for use in V and Q updates
            with torch.no_grad():
                next_state_input = construct_graph(self.inner_node_count, self.inner_edge_types, self.inner_dim,
                                            graph_structure, next_obs, latent)
                target_v_values = self.target_vf(next_state_input)
            '''
            q1_pred = self.qf1(obs, actions, latent, graph_structure=graph_structure)
            q2_pred = self.qf2(obs, actions, latent, graph_structure=graph_structure)
            v_pred = self.vf(obs, detached_latent, graph_structure=graph_structure)
            with torch.no_grad():
                target_v_values = self.target_vf(next_obs, detached_latent, graph_structure=graph_structure)

            # KL constraint on z if probabilistic
            self.context_vector_optimizer.zero_grad()
            self.context_graph_optimizer.zero_grad()
            if self.use_information_bottleneck:
                vector_kl_div, graph_kl_div = self.agent.compute_kl_div()
                vector_kl_loss = self.kl_lambda * vector_kl_div
                vector_kl_loss.backward(retain_graph=True)
                
                graph_kl_loss = self.graph_kl_lambda * graph_kl_div
                graph_kl_loss.backward(retain_graph=True)

            # qf and encoder update (note encoder does not get grads from policy or vf)
            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            # scale rewards for Bellman update
            q_target = rewards * self.reward_scale + (1. - terms) * self.discount * target_v_values
            qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
            qf_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()
            self.context_vector_optimizer.step()
            
            # Update context graph encoder to match results from sim annealing
            #context_graph_loss = torch.sum(-1 * F.one_hot(self.agent.graph_structure, num_classes=self.inner_edge_types) * self.agent.graph_structure_probs.log())
            context_graph_loss = F.cross_entropy((self.agent.graph_structure_probs + 1e-9).flatten(0, -2).log(), self.agent.graph_structure.flatten())
            context_graph_loss.backward()
            self.context_graph_optimizer.step()

            # compute min Q on the new actions
            min_q_new_actions = self._min_q(obs, new_actions, detached_latent, graph_structure)

            # vf update
            v_target = min_q_new_actions - log_pi
            vf_loss = self.vf_criterion(v_pred, v_target.detach())
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            self._update_target_network()

            # policy update
            # n.b. policy update includes dQ/da
            log_policy_target = min_q_new_actions

            policy_loss = (
                    log_pi - log_policy_target
            ).mean()

            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=2).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
        # Update agent sim-anneal temperature to better match current goal acc rate
        self.agent.update_temp(self.sim_anneal_goal_acc_rate)

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['Vector KL Divergence'] = ptu.get_numpy(vector_kl_div)
                self.eval_statistics['Vector KL Loss'] = ptu.get_numpy(vector_kl_loss)
                self.eval_statistics['Graph KL Divergence'] = ptu.get_numpy(graph_kl_div)
                self.eval_statistics['Graph KL Loss'] = ptu.get_numpy(graph_kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Context Graph Encoder Loss'] = ptu.get_numpy(context_graph_loss)
            self.eval_statistics['Train Task Indices'] = indices
            self.eval_statistics['Context Graph Structure'] = ptu.get_numpy(graph_structure)
            self.eval_statistics['SA Goal Accept Rate'] = self.sim_anneal_goal_acc_rate
            
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_vector_encoder=self.agent.context_vector_encoder.state_dict(),
            context_graph_encoder=self.agent.context_graph_encoder.state_dict(),
        )
        return snapshot
