from trainerGym import PPOBuffer
from dictloader import ExperienceDataset, ExperienceSampler
import gym
from gym import spaces
import numpy as np
import torch
from torch.utils.data import DataLoader


env=gym.make('Pong-v0')
space_dict = {}
space_dict['rgb'] = spaces.Box(low=0, high=255, shape=(2, 2, 2))
# space_dict['pointgoal_with_gps_compass'] = env.observation_space
obs_space = spaces.Dict(space_dict)
# end TODO:
act_shape = env.action_space.n
steps = 20
num_rec_layers = 2
hidden_state_size = 10
prev_action = -1
gamma = 0.99
lam = 0.97
buffer = PPOBuffer(obs_space, act_shape, steps, num_rec_layers, hidden_state_size, gamma, lam, n_envs=1)
mask = np.zeros((1,1))
hidden_state = np.ones((num_rec_layers, 1, hidden_state_size))*-1
obs_dict = {'rgb':np.ones((2,2,2))*-1}
obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs_dict.items()}

for i in range(steps):
    next_obs_dict = {'rgb':np.ones((2,2,2))*i}
    action = i
    reward = i
    value = i
    log_prob = i
    next_hidden_state = np.ones((num_rec_layers, 1, hidden_state_size))*i

    buffer.store(obs, action, reward, value, log_prob, hidden_state, mask, prev_action)

    obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in next_obs_dict.items()}
    prev_action = action
    hidden_state = next_hidden_state
    mask = [0.0 if i%7==0 else 1.0]
    prev_action = action
    if i%7==0:
        buffer.finish_path(0)

minibatch_size = 5
data = buffer.get()
dataset = ExperienceDataset(data)
sampler = ExperienceSampler(dataset, minibatch_size, drop_last=True)
data_loader = DataLoader(dataset, batch_size=minibatch_size, sampler=sampler)

for i_batch, minibatch in enumerate(data_loader):
    print(minibatch)
