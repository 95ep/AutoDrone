from risenet.simple_net import SimpleNet
import gym
import torch
from risenet.gymAgent import GymResNetPolicy
from trainerGym import zero_pad_obs
from gym import spaces

env = gym.make('Pong-v0')
space_dict = {}
space_dict['rgb'] = spaces.Box(low=0, high=255, shape=(256, 256, 3))
observation_space = spaces.Dict(space_dict)
ac = GymResNetPolicy(observation_space, env.action_space)
ac.load_state_dict(torch.load('D:/Exjobb2020ErikFilip/AutoDrone/runs/pong-weekend/saved_models/model2000.pth'))

obs =  env.reset()
obs = zero_pad_obs(obs)
obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}
hidden_state = torch.zeros(ac.net.num_recurrent_layers, 1, ac.net.output_size)
action = torch.tensor([0])
mask = torch.zeros(1,1)
nSteps = 0
while 1==1:
    with torch.no_grad():
        _, action, _ , hidden_state = ac.act(
            obs, hidden_state, action, mask, deterministic=False)

    obs, reward, done, _ = env.step(action.item())
    obs = zero_pad_obs(obs)
    obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}
    mask = torch.tensor(
        [0.0] if done else [1.0], dtype=torch.float
    )
    env.render()
    nSteps +=  1
    if done:
        print("Died at step {}".format(nSteps))
        obs =  env.reset()
        obs = zero_pad_obs(obs)
        obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}
        nSteps = 0
