from gym import spaces
import torch
import torch.nn as nn
from .neural_network import PointNavResNetPolicy


def neural_agent(rgb=True, depth=True, gps_compass=True):
    r"""Creates a resnet50 neural network model, that expects
        256x256 pixel visual input + [distance, angle] compass input
        and has output dimension = (4,) - [stop, forward, left, right]

    Args:
        rgb: does the input contain rgb images
        depth: does the input contain depth images
        gps_compass: does the input contain compass data

    Returns:
    """
    space_dict = {}
    if rgb:
        space_dict['rgb'] = spaces.Box(low=0, high=255, shape=(256, 256, 3))

    if depth:
        space_dict['depth'] = spaces.Box(low=0, high=255, shape=(256, 256, 1))

    if gps_compass:
        space_dict['pointgoal_with_gps_compass'] = spaces.Box(low=0, high=1, shape=(2,))

    observation_space = spaces.Dict(space_dict)
    action_space = spaces.Discrete(4)
    return PointNavResNetPolicy(observation_space, action_space)


def load_pretrained_weights(neural_network, path_to_weights):
    weight_dict = torch.load(path_to_weights)["state_dict"]
    weights = {k[len('actor_critic.'):]: v for k, v in weight_dict.items()}
    neural_network.load_state_dict(weights)


def change_action_dim(neural_network, dim_actions):
    neural_network.replace_output_layer(dim_actions)
    neural_network.replace_prev_action_embedding_layer(dim_actions)

