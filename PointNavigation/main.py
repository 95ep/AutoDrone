import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym import spaces

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/Filip/Projects/RISE/AutoDrone/PointNavigation/habitat_baselines')
sys.path.insert(1, 'C:/Users/Filip/Projects/RISE/AutoDrone/PointNavigation')

from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

observation_space = spaces.Dict(
    {#"rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3)),
    "depth": spaces.Box(low=0, high=255, shape=(256, 256, 1)),
    "pointgoal_with_gps_compass" : spaces.Box(low=0, high=1, shape=(2,))}
)
action_space = spaces.Discrete(4)


# Establish skeleton
actor_critic = PointNavResNetPolicy(observation_space, action_space)
#print(actor_critic)
#print(actor_critic.state_dict().keys())


# Load pretrained weights
#weight_directory = 'C:/Users/Filip/Projects/RISE/AutoDrone/PointNavigation/ddppo-models/'
#weight_file = 'gibson-4plus-resnet50.pth' # depth only
#weight = 'gibson-2plus-se-resneXt50-rgb.pth'
#weight_path = weight_directory + weight_file
weight_path = 'habitat_baselines/ddppo-models/gibson-4plus-resnet50.pth'
weight_dict = torch.load(weight_path)["state_dict"] # "model_args" contain additional info
# Remove "actor_critic." from keys to align names
weights = {k[len('actor_critic.') :]: v for k, v in weight_dict.items()}
actor_critic.load_state_dict(weights)

# , self.goal_sensor_uuid,

depth_observation = torch.zeros(256,256,1).unsqueeze(0)
target_observation = torch.FloatTensor([1,0]).unsqueeze(0)
observations = {
    "depth": depth_observation,
    "pointgoal_with_gps_compass": target_observation
}

prev_actions = torch.tensor([0])
#h0 = torch.zeros(actor_critic.net._hidden_size)
#c0 = torch.zeros(actor_critic.net._hidden_size)
#rnn_hidden_states = actor_critic.net.state_encoder._pack_hidden([c0,h0])
rnn_hidden_states = torch.zeros(actor_critic.net.num_recurrent_layers,
            1,actor_critic.net._hidden_size)
#print("Dim rnn hidden: " + str(rnn_hidden_states.shape))
#print("unpacked: " + str(actor_critic.net.state_encoder._unpack_hidden(rnn_hidden_states)[1].shape))
masks = torch.zeros(1,1)

value, action, action_log_probs, rnn_hidden_states = actor_critic.act(
             observations,
             rnn_hidden_states,
             prev_actions,
             masks,
             deterministic=False,
         )
print("value: " + str(value.item()))
print("action: " + str(action.item()))
