import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import airsim
import gym
from gym import spaces
import time
import sys
from agent import *

# sys.path.insert(1, 'C:/Users/Filip/Projects/RISE/AutoDrone/PointNavigation/habitat_baselines')
# sys.path.insert(1, 'C:/Users/Filip/Projects/RISE/AutoDrone/PointNavigation')
sys.path.insert(1, 'D:/Exjobb2020ErikFilip/AutoDrone/PointNavigation/habitat_baselines')
sys.path.insert(1, 'D:/Exjobb2020ErikFilip/AutoDrone/PointNavigation')

from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

# ================ LOAD NEURAL NETWORK ========================

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
weight_path = 'habitat_baselines/ddppo-models/gibson-2plus-resnet50.pth'
weight_dict = torch.load(weight_path)["state_dict"] # "model_args" contain additional info
# Remove "actor_critic." from keys to align names
weights = {k[len('actor_critic.') :]: v for k, v in weight_dict.items()}
actor_critic.load_state_dict(weights)
actor_critic.eval()

# ===================== TEST NET OUTPUT ==============================
"""
depth_observation = torch.rand(256,256,1).unsqueeze(0)
target_observation = torch.FloatTensor([1,0]).unsqueeze(0)
observations = {
    "depth": depth_observation,
    "pointgoal_with_gps_compass": target_observation
}

prev_actions = torch.tensor([0])
rnn_hidden_states = torch.zeros(actor_critic.net.num_recurrent_layers,
            1,actor_critic.net._hidden_size)
masks = torch.ones(1,1)

# actions: [stop, forward (0.25m), left (10deg), right (10deg)]
value, action, action_log_probs, rnn_hidden_states = actor_critic.act(
             observations,
             rnn_hidden_states,
             prev_actions,
             masks,
             deterministic=False,
         )
print("value: " + str(value.item()))
print("action: " + str(action.item()))
"""

# ==================== AIRSIM STUFF ============================



def printInfo(currentClient):
    pos = currentClient.simGetGroundTruthKinematics().position
    q = currentClient.simGetGroundTruthKinematics().orientation
    yawDeg = airsim.to_eularian_angles(q)[2]/np.pi*180

    print('Current yaw is {} deg'.format(yawDeg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))


def targetPositionToCompass(currentClient, target):

    pos = currentClient.simGetGroundTruthKinematics().position
    q = currentClient.simGetGroundTruthKinematics().orientation
    yaw_rad = airsim.to_eularian_angles(q)[2]
    target_vec = np.array([target[0] - pos.x_val, target[1] - pos.y_val])
    if target_vec[0] == 0 and target_vec[1] == 0:
        return np.array([0, 0])

    u = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
    v = target_vec / np.linalg.norm(target_vec)
    angle_mag = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    angle_sign = np.sign(np.cross(u, v))
    angle_sign = 1 if angle_sign == 0 else angle_sign
    return torch.tensor([np.linalg.norm(target_vec), angle_mag * angle_sign])

def generateTarget(currentClient):
    pos = currentClient.simGetGroundTruthKinematics().position
    x = np.random.rand()*40 -20.0 + pos.x_val
    y = np.random.rand()*40 -20.0 + pos.y_val
    print("Generating new target at location ({}, {})".format(x,y))
    return np.array([x,y])


# =================== TEST IN AIRSIM ===========================

# connect to the AirSim simulator


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Wait a little while to make sure info is correct
time.sleep(1)
printInfo(client)

previous_action = torch.tensor([1])
rnn_hidden_states = torch.zeros(actor_critic.net.num_recurrent_layers,
            1, actor_critic.net._hidden_size)
masks = torch.ones(1,1)

target = generateTarget(client)
print("===============================================================")
for i in range(500):

    rgb_observation, depth_observation = getImages(client) #torch.from_numpy(getImage())
    target_position = targetPositionToCompass(client, target)
    target_observation = target_position.unsqueeze(0)
    distance = target_position[0]

    observations = {
        "depth": torch.from_numpy(depth_observation).unsqueeze(2).unsqueeze(0),
        "pointgoal_with_gps_compass": target_observation
    }

    with torch.no_grad():
        value, action, action_log_probs, rnn_hidden_states = actor_critic.act(
            observations,
            rnn_hidden_states,
            previous_action,
            masks,
            deterministic=False,
        )
    print('Distance and Angle to target: [{}, {}],     value: {}'.format(distance, target_position[1]/np.pi * 180, value.item()))
    # Act in environment
    if action == 0:
        print('Agent predicts that it has reached the goal. Do nothing')
        if distance <= 0.2:
            print('Prediction is accurate.')
        else:
            print('Prediction is false. Force the agent to move forward')
            moveForward(client)
    elif action == 1:
        print('Moving forward')
        moveForward(client)
    elif action == 2:
        print('Rotating left')
        rotateLeft(client)
    elif action == 3:
        print('Rotating right')
        rotateRight(client)

    previous_action = action

    if distance <= 0.2:
        print("Target reached!")
        target = generateTarget(client)
    print("===============================================================")
