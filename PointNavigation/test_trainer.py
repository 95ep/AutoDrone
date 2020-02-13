from trainer_cartpole import PPOBuffer
import numpy as np

from gym import spaces

obs_space = spaces.Box(0,0,(4,))
act_shape = 1
steps = 30
num_rec_layers = 1
hidden_state_size = 1
gamma = 0.99
lam = 0.97

buff = PPOBuffer( obs_space, act_shape, steps, gamma, lam)

obs = np.zeros((4,))

prev_act = -1
for i in range(steps):
    act = i
    next_obs = np.ones((1,4))*i
    rew = 1
    val = 1
    logp = -0.5
    done = i == 20

    buff.store(obs, act, rew, val, logp)
    if done:
        buff.finish_path(last_val=0)

    prev_act = act
    obs = next_obs


buff.finish_path(last_val=5)

data = buff.get()
returns = data['ret']

obs = data['obs']
print('print obs')
for o in obs:
    print(o)


print('print return')
for ret in returns:
    print(ret)

advantages = data['adv']
print('print adv')
for adv in advantages:
    print(adv)
