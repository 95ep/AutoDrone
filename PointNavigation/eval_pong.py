import gym
import torch
from risenet.neutral_net import NeutralNet
from trainer_new import process_obs
from gym import spaces
from gym.wrappers.frame_stack import FrameStack

env_str = "Atari"
parameters = {'training': {'height':128, 'width':128}}

env = gym.make('PongDeterministic-v4')
env = FrameStack(env, 4)
vector_encoder, visual_encoder, compass_encoder = False, False, False
vector_shape, visual_shape, compass_shape = None, None, None
n_actions = env.action_space.n
visual_encoder = True
visual_shape = (128, 128, 3*4)


ac = NeutralNet(has_vector_encoder=vector_encoder, vector_input_shape=vector_shape,
                has_visual_encoder=visual_encoder, visual_input_shape=visual_shape,
                has_compass_encoder=compass_encoder, compass_input_shape=compass_shape,
                num_actions=n_actions, has_previous_action_encoder=False,
                hidden_size=32, num_hidden_layers=1)
ac.load_state_dict(torch.load('D:/Exjobb2020ErikFilip/AutoDrone/runs/neutral-pong-weekend/saved_models/model900.pth'))

obs, episode_return, episode_len = env.reset(), 0, 0
obs_vector, obs_visual, obs_compass = process_obs(obs, env_str, parameters)
comb_obs = tuple(o for o in [obs_vector, obs_visual, obs_compass] if o is not None)


done = False
nSteps = 0
ret = 0
while not done:
    with torch.no_grad():
        _, action, _ = ac.act(comb_obs, deterministic=True)
    next_obs, reward, done, _ = env.step(action.item())
    env.render()
    nSteps += 1
    ret += reward
    obs_vector, obs_visual, obs_compass = process_obs(next_obs, env_str, parameters)
    comb_obs = tuple(o for o in [obs_vector, obs_visual, obs_compass] if o is not None)

    if done:
        print("Died at step {}".format(nSteps))

print("Return {}".format(ret))
