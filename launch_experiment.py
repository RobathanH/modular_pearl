"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    
    print("obs_dim:", obs_dim, "\naction_dim:", action_dim)

    # instantiate networks
    if variant['graph_pearl']:
        assert variant['algo_params']['use_information_bottleneck'], "Only implemented for info bottleneck enabled"
        from graph_pearl import GraphModule, Graph_TanhGaussianPolicy, Graph_PEARLAgent, Graph_PEARLSoftActorCritic
        
        gnn_node_count = variant['algo_params']['gnn_node_count']
        gnn_edge_types = variant['algo_params']['gnn_edge_types']
        pre_gnn_fc_layers = variant['algo_params']['pre_gnn_fc_layers']
        gnn_layers = variant['algo_params']['gnn_layers']
        post_gnn_fc_layers = variant['algo_params']['post_gnn_fc_layers']
        split_layer_into_nodes = variant['algo_params'].get('split_layer_into_nodes', False)
        
        latent_dim = variant['latent_size']
        net_size = variant['net_size']
        
        encoder_model = RecurrentEncoder if variant['algo_params']['recurrent'] else MlpEncoder
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
        context_vector_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
        if latent_dim == 0:
            context_vector_encoder_output_dim = 2
        
        context_vector_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_vector_encoder_output_dim
        )
        context_graph_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=(gnn_node_count**2) * gnn_edge_types
        )
        
        
        
        def get_layer_size_list(arg):
            if isinstance(arg, int):
                return [net_size] * arg
            elif isinstance(arg, list):
                return arg
            else:
                raise NotImplementedError
        
        gnn_layer_sizes = get_layer_size_list(gnn_layers)
        pre_gnn_layer_sizes = get_layer_size_list(pre_gnn_fc_layers)
        post_gnn_layer_sizes = get_layer_size_list(post_gnn_fc_layers)
        
        qf1 = GraphModule(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            gnn_edge_types=gnn_edge_types,
            gnn_node_count=gnn_node_count,
            gnn_layer_sizes=gnn_layer_sizes,
            pre_gnn_layer_sizes=pre_gnn_layer_sizes,
            post_gnn_layer_sizes=post_gnn_layer_sizes,
            split_layer_into_nodes=split_layer_into_nodes
        )
        qf2 = GraphModule(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            gnn_edge_types=gnn_edge_types,
            gnn_node_count=gnn_node_count,
            gnn_layer_sizes=gnn_layer_sizes,
            pre_gnn_layer_sizes=pre_gnn_layer_sizes,
            post_gnn_layer_sizes=post_gnn_layer_sizes,
            split_layer_into_nodes=split_layer_into_nodes
        )
        vf = GraphModule(
            input_size=obs_dim + latent_dim,
            output_size=1,
            gnn_edge_types=gnn_edge_types,
            gnn_node_count=gnn_node_count,
            gnn_layer_sizes=gnn_layer_sizes,
            pre_gnn_layer_sizes=pre_gnn_layer_sizes,
            post_gnn_layer_sizes=post_gnn_layer_sizes,
            split_layer_into_nodes=split_layer_into_nodes
        )
        policy = Graph_TanhGaussianPolicy(
            obs_dim,
            latent_dim,
            action_dim,
            gnn_edge_types=gnn_edge_types,
            gnn_node_count=gnn_node_count,
            gnn_layer_sizes=gnn_layer_sizes,
            pre_gnn_layer_sizes=pre_gnn_layer_sizes,
            post_gnn_layer_sizes=post_gnn_layer_sizes,
            split_layer_into_nodes=split_layer_into_nodes
        )
        agent = Graph_PEARLAgent(
            latent_dim,
            context_vector_encoder,
            context_graph_encoder,
            policy,
            qf1,
            qf2,
            vf,
            **variant['algo_params']
        )
        algorithm = Graph_PEARLSoftActorCritic(
            env=env,
            train_tasks=list(tasks[:variant['n_train_tasks']]),
            eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
            agent=agent,
            latent_dim=latent_dim,
            **variant['algo_params']
        )
        
        # optionally load pre-trained weights
        if variant['path_to_weights'] is not None:
            path = variant['path_to_weights']
            context_vector_encoder.load_state_dict(torch.load(os.path.join(path, 'context_vector_encoder.pth')))
            context_graph_encoder.load_state_dict(torch.load(os.path.join(path, 'context_graph_encoder.pth')))
            qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
            qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
            vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
            # TODO hacky, revisit after model refactor
            algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
            policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
        
    
    
    else:
        # Remove non-graph-pearl algo_params
        graph_only_algo_params = [
            "gnn_node_count",
            "gnn_edge_types",
            "pre_gnn_fc_layers",
            "gnn_layers",
            "post_gnn_fc_layers",
            "bouncegrad_iterations",
            "context_graph_lr",
            "graph_kl_lambda",
            "sim_anneal_train_proposals",
            "sim_anneal_eval_proposals",
            "sim_anneal_init_temp",
            "sim_anneal_init_goal_acc_rate",
            "sim_anneal_final_goal_acc_rate",
            "persistent_task_graph_structures"
        ]
        for key in graph_only_algo_params:
            if key in variant['algo_params']:
                del variant['algo_params'][key]
        
        latent_dim = variant['latent_size']
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
        context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
        net_size = variant['net_size']
        recurrent = variant['algo_params']['recurrent']
        encoder_model = RecurrentEncoder if recurrent else MlpEncoder

        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
        )
        qf1 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
        )
        qf2 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
        )
        vf = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + latent_dim,
            output_size=1,
        )
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            init_w=1e-2
        )
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
        )
        algorithm = PEARLSoftActorCritic(
            env=env,
            train_tasks=list(tasks[:variant['n_train_tasks']]),
            eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
            nets=[agent, qf1, qf2, vf],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

        # optionally load pre-trained weights
        if variant['path_to_weights'] is not None:
            path = variant['path_to_weights']
            context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
            qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
            qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
            vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
            # TODO hacky, revisit after model refactor
            algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
            policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))



    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()

