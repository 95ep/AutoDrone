import torch
from torch.optim import Adam
import time
import numpy as np


'''
def _compute_loss_pi(data, actor_critic, clip_ratio, logger):
    # TODO: perhaps merge _compute_loss_pi and _compute_loss_v to single function.
    obs, act, adv, logp_old, hidden = data['obs'], data['act'], data['adv'], data['logp'], data['hidden']

    # Policy loss
    # TODO: make sure our network obeys this.
    # This line makes a call to forward of torch.nn.module .pi
    # Returns pi: action (Normal) distributions for given obs and logp: log likelihood of action
    # under those distributions
    pi, logp = actor_critic.pi(obs, act)

    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # TODO: Add some logging here

    return loss_pi


def _compute_loss_v(data, actor_critic):
    obs, ret = data['obs'], data['ret']
    # TODO: make sure network obeys this
    return ((actor_critic.v(obs) - ret) ** 2).mean()
'''


def _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef):
    obs, act, ret, adv, logp_old, hidden, prev_actions, masks = data['obs'], data['act'], data['ret'], \
                                                                data['adv'], data['logp'], data['hidden'],\
                                                                data['prev_action'], data['masks']

    # Is dist_entropy == KL divergence?
    # How is prev_actions and masks used?
    #   Mask comes from RNNStateEncoder part of the network. Think it is used to
    #   hide hidden state during forward pass.
    #   From what I can see in the code prev_actions are actually never used but required argument for forwards passes
    # How does .evaluate_actions() work?
    values, logp, dist_entropy, _ = actor_critic.evaluate_actions(obs, hidden, prev_actions, masks, act)

    # Calc ratio of logp
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    action_loss = -torch.min(surr1, surr2).mean()

    #TODO: Facebook put factor of 0.5 infront of value_loss. Not sure why so I have omitted.
    value_loss = (ret - values).pow(2).mean()

    total_loss = value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef
    return total_loss


def _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, logger, value_loss_coef, entropy_coef):
    # Spinning up used target_kl for early stopping. Is it smart thing to include?
    data = buffer.get()

    for i in range(train_iters):
        optimizer.zero_grad()
        total_loss = _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef)
        total_loss.backward()
        optimizer.step()

    # TODO: add some logging of this

class Logger:
    # TODO: Implement logger class. Probably generally useful so probably placed in some util module
    def __init__(self):
        pass

    def store(self):
        pass


class PPOBuffer:
    # Stores trajectories collected.
    # Able to store: obs, action, reward, value, log_prob, hidden_state
    # TODO - implement this class and its functions

    def __init__(self):
        pass

    def store(self, obs, act, rew, val, logp, hidden_state):
        pass

    def finish_path(self, last_val=0):
        # last_val = 0 if agent died, otherwise V(s_T)
        pass

    def get(self):
        # return all data
        pass


def PPO_trainer(env, actor_critic, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99,
                clip_ratio=0.2, lr=3e-4,  train_iters=80, lam=0.97, max_episode_len=1000,
                value_loss_coef=1, entropy_coef=0.01, logger_kwargs=dict(), save_freq=10):
    # value_loss_coef and entropy_coef taken from https://arxiv.org/abs/1707.06347
    # TODO: check input parameters and their descriptions
    """
    :param env: environment object. Should fulfill OpenAI Gym API
    :param actor_critic: PyTorch Module TODO: Define how interactions with it should look
    :param seed: Random seed, do we need it?
    :param steps_per_epoch:
    :param epochs:
    :param gamma: Discount factor, between 0 and 1
    :param clip_ratio:
    :param lr:
    :param train_iters:
    :param lam: Lambda for Generalized Advantage Estimation (GAE). Used in buffer to calc advantages
                of state-action pairs
    :param max_ep_len:
    :param logger_kwargs:
    :param save_freq: How often to save policy and value functions
    :return: Trained network. #TODO: No return statement needed right?
    """

    # Set up logger
    logger  = Logger()

    # Seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up env

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Count variables. Something they do and add to logg in spinningUp but not necessary

    # Set up experience buffer
    buffer = PPOBuffer()

    # This is just copied from baseline. My understanding is that filter passes each parameter to the lambda function
    # and then creates a list of alla parameters for which .requires_grad is True
    optimizer =  Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())), lr=lr)

    # Set up model saving to logger
    # TODO: Implement model saving

    # Prepare for interaction with env
    start_time = time.time()
    obs, episode_return, episode_len = env.reset(), 0, 0


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            # TODO: Determine format of input to actor_critic
            action, value, log_prob, hidden_state = actor_critic.step(obs) # feed init obs

            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            episode_len += 1

            # Save and log
            buffer.store(obs, action, reward, value, log_prob, hidden_state)
            logger.store()

            # Update obs
            obs  = next_obs

            timeout = episode_len == max_episode_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning, trajectory cut off by epoch at {} steps.'.format(episode_len), flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                # TODO: not 100 % sure what this mean and the effect of it
                if timeout or epoch_ended:
                    _, value, _, _ = actor_critic.step(obs)
                else:
                    value = 0
                buffer.finish_path(value)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=episode_return, EpLen=episode_len)
                obs, episode_return, episode_len = env.reset(), 0, 0

        # A epoch of experience is collected
        # Save model
        # TODO - add code for this

        # Perform PPO update
        _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, logger, value_loss_coef, entropy_coef)

        # Log info about epoch
        # TODO: add code for this