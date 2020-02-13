from risenet.simple_net import SimpleNet
import gym
import torch

env = gym.make('CartPole-v0')
ac = SimpleNet(*env.observation_space.shape, env.action_space.n, hidden_size=64)
ac.load_state_dict(torch.load('D:/Exjobb2020ErikFilip/AutoDrone/runs/cartpole-cpuMightWork/saved_models/model500.pth'))

obs =  torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
nSteps =0
while 1==1:
    with torch.no_grad():
        _, action, _ = ac.act(obs)

    obs, reward, done, _ = env.step(action.item())
    obs =  torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    env.render()
    nSteps +=  1
    if done:
        print("Died at step {}".format(nSteps))
        obs =  torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        nSteps = 0
