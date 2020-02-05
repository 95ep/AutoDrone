import torch
from torch.optim import Adam
import time
import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    # Helper function. Some magic from scipy.
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef):
    obs, act, ret, adv, logp_old, hidden, prev_actions, masks = data['obs'], data['act'], data['ret'], \
                                                                data['adv'], data['logp'], data['hidden'],\
                                                                data['prev_act'], data['mask']

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

    # TODO: Facebook put factor of 0.5 infront of value_loss. Not sure why so I have omitted.
    value_loss = (ret - values).pow(2).mean()

    total_loss = value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef
    return total_loss


def _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, value_loss_coef, entropy_coef):
    # Spinning up used target_kl for early stopping. Is it smart thing to include?
    data = buffer.get()

    for i in range(train_iters):
        optimizer.zero_grad()
        total_loss = _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef)
        total_loss.backward()
        optimizer.step()


class Logger:
    # TODO: Implement logger class. Probably generally useful so probably placed in some util module
    def __init__(self):
        pass

    def store(self):
        pass


class PPOBuffer:
    # Stores trajectories collected.
    def __init__(self, obs_space, act_shape, steps, num_rec_layers, hidden_state_size, gamma, lam, n_envs=1):
        self.obs_buf = {}
        for sensor in obs_space.spaces:
            # * operator unpacks arguments, I think
            self.obs_buf[sensor] = torch.zeros(steps, n_envs, *obs_space.spaces[sensor].shape)

        self.act_buf = np.zeros(steps, act_shape)
        self.prev_act_buf = np.zeros(steps, act_shape)
        self.hidden_buf = np.zeros((steps, num_rec_layers, n_envs, hidden_state_size), dtype=np.float32)
        self.rew_buf = np.zeros(steps, dtype=np.float32)
        self.adv_buf = np.zeros(steps, dtype=np.float32)
        self.ret_buf = np.zeros(steps, dtype=np.float32)
        self.val_buf = np.zeros(steps, dtype=np.float32)
        self.logp_buf = np.zeros(steps, dtype=np.float32)
        self.mask_buf = np.zeros(steps, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0    # Keeps track of number of interaction in buffer
        self.path_start_idx = 0
        self.max_size = steps

    def store(self, obs, act, rew, val, logp, hidden_state, mask, prev_act):
        """
        Add one step of interactions to the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.hidden_buf = hidden_state
        self.mask_buf = mask
        self.prev_act_buf = prev_act
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)[:-1]

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        # return all data in the form och torch tensors
        assert self.ptr == self.max_size    # buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        # Advantage normalization
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    hidden=self.hidden_buf, mask=self.mask_buf, prev_act=self.prev_act_buf)

        # Make into torch tensors
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data}

        # Add obs manually since itself is a dict
        data['obs'] = self.obs_buf
        return data


def PPO_trainer(env, actor_critic, num_rec_layers, hidden_state_size, seed=0, steps_per_epoch=4000,
                epochs=50, gamma=0.99, clip_ratio=0.2, lr=3e-4,  train_iters=80, lam=0.97,
                max_episode_len=1000, value_loss_coef=1, entropy_coef=0.01, logger_kwargs=dict(),
                save_freq=10):
    # value_loss_coef and entropy_coef taken from https://arxiv.org/abs/1707.06347
    # TODO: check input parameters and their descriptions
    """
    :param env: environment object. Should fulfill OpenAI Gym API
    :param actor_critic: PyTorch Module
    :param num_rec_layers:
    :param hidden_state_size:
    :param seed: Random seed, do we need it?
    :param steps_per_epoch:
    :param epochs:
    :param gamma: Discount factor, between 0 and 1
    :param clip_ratio:
    :param lr:
    :param train_iters:
    :param lam: Lambda for Generalized Advantage Estimation (GAE). Used in buffer to calc advantages
                of state-action pairs
    :param max_episode_len:
    :param value_loss_coef:
    :param entropy_coef:
    :param logger_kwargs:
    :param save_freq: How often to save policy and value functions
    :return: Trained network.
    """

    # Set up logger
    logger = Logger()

    # Seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get some env variables
    obs_space = env.observation_space
    act_shape = env.action_space.shape

    # Count variables. Something they do and add to logg in spinningUp but not necessary

    # Set up experience buffer
    buffer = PPOBuffer(obs_space, act_shape, steps_per_epoch, num_rec_layers, hidden_state_size, gamma, lam,)

    # This is just copied from baseline. My understanding is that filter passes each parameter to the lambda function
    # and then creates a list of alla parameters for which .requires_grad is True
    optimizer = Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())), lr=lr)

    # Set up model saving to logger
    # TODO: Implement model saving

    # Prepare for interaction with env
    start_time = time.time()
    obs, episode_return, episode_len = env.reset(), 0, 0
    # Shape of hidden state is (n_rec_layers, num_envs, recurrent_hidden_state_size).
    # should be able to access these from PointNavResNetNet properties
    hidden_state = torch.zeros(actor_critic.num_recurrent_layers(), 1, actor_critic.output_size())
    prev_action = torch.zeros(1)
    mask = torch.zeros(1)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            with torch.no_grad():
                action, value, log_prob, hidden_state = actor_critic.act(
                                                      obs, hidden_state, prev_action, mask)

            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            episode_len += 1

            mask = torch.tensor(
                0.0 if done else 1.0, dtype=torch.float
            )

            # Save to buffer and log
            buffer.store(obs, action, reward, value, log_prob, hidden_state, mask, prev_action)
            logger.store()

            # Update obs and prev_action
            obs = next_obs
            prev_action = action

            timeout = episode_len == max_episode_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning, trajectory cut off by epoch at {} steps.'.format(episode_len), flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _, _ = actor_critic.step(obs)
                else:
                    value = 0
                buffer.finish_path(value)
                if terminal:
                    pass
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=episode_return, EpLen=episode_len)
                # Reset if episode ended
                obs, episode_return, episode_len = env.reset(), 0, 0

        # A epoch of experience is collected
        # Save model
        # TODO - add code for this

        # Perform PPO update
        _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, value_loss_coef, entropy_coef)

        # Log info about epoch
        # TODO: add code for this
