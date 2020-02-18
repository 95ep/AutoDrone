import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dictloader import ExperienceDataset, ExperienceSampler
import time
import numpy as np
import scipy.signal
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_pad_obs(obs, width=256, height=256, channels=3):
    padded = np.zeros((width, height, channels))
    w_idx = (width-obs.shape[1]) // 2
    h_idx = (height-obs.shape[0]) // 2
    padded[h_idx:obs.shape[0]+h_idx, w_idx:obs.shape[1]+w_idx, :] = obs

    obs_dict = {'rgb':padded}
    return obs_dict

def discount_cumsum(x, discount):
    # Helper function. Some magic from scipy.
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef):
    obs, act, ret, adv, logp_old, hidden, prev_actions, masks = data['obs'], data['act'], data['ret'], \
                                                                data['adv'], data['logp'], data['hidden'], \
                                                                data['prev_act'], data['mask']

    for k,v in obs.items():
        obs[k]=v.to(device=device)
    #     print("Shape of obs {} is {}".format(k, obs[k].shape))
    act = act.to(device=device)
    ret = ret.to(device=device)
    adv = adv.to(device=device)
    logp_old = logp_old.to(device=device)
    hidden = hidden.to(device=device)
    prev_actions = prev_actions.to(device=device)
    masks = masks.to(device=device)


    # print("Shape of action: {}".format(act.shape))
    # print("Shape of prev_actions {}".format(prev_actions.shape))
    values, logp, dist_entropy, _ = actor_critic.evaluate_actions(obs, hidden[0], prev_actions, masks, act)
    # values = values.squeeze()
    # logp = logp.squeeze()

    # Calc ratio of logp
    # print("Shape of logp: {}".format(logp.shape))
    # print("Shape of logp_old: {}".format(logp_old.shape))
    ratio = torch.exp(logp - logp_old)
    # print("Shape of ratio: {}".format(ratio.shape))
    # print("Shape of adv: {}".format(adv.shape))
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # print("Shape of surr1: {}".format(surr1.shape))
    # print("Shape of surr2: {}".format(surr2.shape))
    action_loss = -torch.min(surr1, surr2).mean()

    # print("Shape of ret: {}".format(ret.shape))
    # print("Shape of values: {}".format(values.shape))
    # print(" (ret - values).pow(2) {}".format( (ret - values).pow(2).shape))
    value_loss = (ret - values).pow(2).mean()

    # print("Shape of entropy: {}".format(dist_entropy.shape))
    total_loss = value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef
    # print("TotalLoss {}".format(total_loss.shape))
    # print(total_loss)

    # Compute approx kl for logging purposes
    approx_kl = (logp_old-logp).mean().item()

    return total_loss, action_loss, value_loss, dist_entropy, approx_kl


def _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, value_loss_coef, entropy_coef, target_kl, minibatch_size):
    # Spinning up used target_kl for early stopping. Is it smart thing to include?
    data = buffer.get()
    total_loss_in_epoch = []
    action_loss_in_epoch = []
    value_loss_in_epoch = []
    entropy_in_epoch = []

    dataset = ExperienceDataset(data)
    sampler = ExperienceSampler(dataset, minibatch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_size=minibatch_size, sampler=sampler)

    for i in range(train_iters):
        optimizer.zero_grad()
        approx_kl_iter = []
        for i_batch, minibatch in enumerate(data_loader):
            total_loss, action_loss, value_loss, entropy, approx_kl = _total_loss(minibatch, actor_critic, clip_ratio,
                                                                   value_loss_coef, entropy_coef)

            total_loss_in_epoch.append(total_loss.item())
            action_loss_in_epoch.append(action_loss.item())
            value_loss_in_epoch.append(value_loss.item())
            entropy_in_epoch.append(entropy.item())
            approx_kl_iter.append(approx_kl)

            total_loss.backward()
            optimizer.step()

        if np.array(approx_kl_iter).mean() > target_kl:
           print("Early stopping at step {} due to reaching max kl.".format(i))
           break


    mean_total_loss = np.mean(np.array(total_loss_in_epoch))
    mean_action_loss = np.mean(np.array(action_loss_in_epoch))
    mean_value_loss = np.mean(np.array(value_loss_in_epoch))
    mean_entropy = np.mean(np.array(entropy_in_epoch))

    return mean_total_loss, mean_action_loss, mean_value_loss, mean_entropy, approx_kl_iter.mean()


class PPOBuffer:
    # Stores trajectories collected.
    def __init__(self, obs_space, act_shape, steps, num_rec_layers, hidden_state_size, gamma, lam, n_envs=1):
        self.obs_buf = {}
        for sensor in obs_space.spaces:
            # * operator unpacks arguments, I think
            self.obs_buf[sensor] = np.zeros((steps, *obs_space.spaces[sensor].shape), dtype=np.float32)
        self.act_buf = np.zeros((steps,1), dtype=np.int)
        self.prev_act_buf = np.zeros((steps,1), dtype=np.int)
        self.hidden_buf = np.zeros((steps, num_rec_layers, n_envs, hidden_state_size), dtype=np.float32)
        self.rew_buf = np.zeros(steps, dtype=np.float32)
        self.adv_buf = np.zeros((steps,1), dtype=np.float32)
        self.ret_buf = np.zeros((steps,1), dtype=np.float32)
        self.val_buf = np.zeros(steps, dtype=np.float32)
        self.logp_buf = np.zeros((steps, 1), dtype=np.float32)
        self.mask_buf = np.zeros((steps, 1), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0  # Keeps track of number of interaction in buffer
        self.path_start_idx = 0
        self.max_size = steps

    def store(self, obs, act, rew, val, logp, hidden_state, mask, prev_act):
        """
        Add one step of interactions to the buffer
        """
        assert self.ptr < self.max_size
        for k,v in obs.items():
            self.obs_buf[k][self.ptr] =  v
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.hidden_buf[self.ptr] = hidden_state
        self.mask_buf[self.ptr] = mask
        self.prev_act_buf[self.ptr] = prev_act
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
        self.adv_buf[path_slice] = np.expand_dims(discount_cumsum(deltas, self.gamma * self.lam), axis=1)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = np.expand_dims(discount_cumsum(rews, self.gamma)[:-1], axis=1)

        self.path_start_idx = self.ptr

    def get(self):
        # return all data in the form och torch tensors
        assert self.ptr == self.max_size  # buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        # Advantage normalization
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    hidden=self.hidden_buf, mask=self.mask_buf, prev_act=self.prev_act_buf)

        # Make into torch tensors
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

        # Add obs manually since itself is a dict
        data['obs'] = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in  self.obs_buf.items()}
        return data


def PPO_trainer(env, actor_critic, num_rec_layers, hidden_state_size, seed=0, steps_per_epoch=4000,
                epochs=50, minibatch_size=64 ,gamma=0.99, clip_ratio=0.2, lr=3e-4, train_iters=80, lam=0.97,
                max_episode_len=1000, value_loss_coef=1, entropy_coef=0.01, target_kl=0.01, save_freq=10,
                log_dir='runs'):
    # value_loss_coef and entropy_coef taken from https://arxiv.org/abs/1707.06347

    # Set up logger
    log_writer = SummaryWriter(log_dir=log_dir + 'log/')

    # Seed torch and numpy
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # Get some env variables
    #obs_space = env.observation_space
    # TODO: hard coded fix
    from gym import spaces
    # space_dict['rgb'] = spaces.Box(low=0, high=255, shape=(256, 256, 3))
    space_dict['pointgoal_with_gps_compass'] = env.observation_space
    obs_space = spaces.Dict(space_dict)
    # end TODO:
    act_shape = env.action_space.n

    # Count variables. Something they do and add to logg in spinningUp but not necessary

    # Set up experience buffer
    buffer = PPOBuffer(obs_space, act_shape, steps_per_epoch, num_rec_layers, hidden_state_size, gamma, lam, )

    # This is just copied from baseline. My understanding is that filter passes each parameter to the lambda function
    # and then creates a list of alla parameters for which .requires_grad is True
    optimizer = Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())), lr=lr)

    # Prepare for interaction with env
    start_time = time.time()
    obs, episode_return, episode_len = env.reset(), 0, 0
    # obs = zero_pad_obs(obs)
    obs = {"pointgoal_with_gps_compass":obs}
    obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}
    # Shape of hidden state is (n_rec_layers, num_envs, recurrent_hidden_state_size).
    # should be able to access these from PointNavResNetNet properties
    hidden_state = torch.zeros(actor_critic.net.num_recurrent_layers, 1, actor_critic.net.output_size)
    prev_action = torch.tensor([0])
    mask = torch.zeros(1,1)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        print('Epoch {} started'.format(epoch))
        # list of episode returns and episode lengths to put to logg
        episode_returns_epoch = []
        episode_len_epoch = []
        for t in range(steps_per_epoch):
            with torch.no_grad():
                value, action, log_prob, next_hidden_state = actor_critic.act(
                    obs, hidden_state, prev_action, mask)

            next_obs, reward, done, _ = env.step(action.item())
            # env.render()
            # next_obs = zero_pad_obs(next_obs)
            next_obs = {"pointgoal_with_gps_compass":next_obs}

            episode_return += reward
            episode_len += 1


            # Save to buffer
            # for k,v in obs.items():
            #     print("Shape of obs {} is {}".format(k, v.shape))
            # print("Shape of action: {}, reward: {}, value: {}".format(action.shape, reward, value.shape))
            # print("... logp: {}, hidden_state: {}, mask: {}, prev_act, {}".format(log_prob.shape, hidden_state.shape, mask.shape, prev_action.shape))
            buffer.store(obs, action, reward, value, log_prob, hidden_state, mask, prev_action)

            # Update obs and prev_action
            obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in next_obs.items()}
            prev_action = action
            hidden_state = next_hidden_state

            timeout = episode_len == max_episode_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            mask = torch.tensor(
                [0.0] if terminal or epoch_ended else [1.0], dtype=torch.float
            )

            if terminal or epoch_ended:
                # if epoch_ended and not terminal:
                #     print('Warning, trajectory cut off by epoch at {} steps.'.format(episode_len), flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    with torch.no_grad():
                        value = actor_critic.get_value(obs, hidden_state, prev_action, mask)
                else:
                    value = 0
                buffer.finish_path(value)

                episode_returns_epoch.append(episode_return)
                episode_len_epoch.append(episode_len)
                # Reset if episode ended
                obs, episode_return, episode_len = env.reset(), 0, 0
                # obs = zero_pad_obs(obs)
                obs = {"pointgoal_with_gps_compass":obs}
                obs = {k:torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}

        # A epoch of experience is collected
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            torch.save(actor_critic.state_dict(), log_dir + 'saved_models/model{}.pth'.format(epoch))

        # Perform PPO update
        actor_critic = actor_critic.to(device=device)
        mean_loss_total, mean_loss_action, mean_loss_value, mean_entropy, approx_kl = _update(actor_critic, buffer,
                                                                                  train_iters, optimizer, clip_ratio,
                                                                                    value_loss_coef, entropy_coef, target_kl, minibatch_size)

        actor_critic = actor_critic.cpu()
        # Calc metrics and log info about epoch
        episode_return_mean = np.mean(np.array(episode_returns_epoch))
        episode_len_mean = np.mean(np.array(episode_len_epoch))

        # Total env interactions:
        log_writer.add_scalar('Progress/TotalEnvInteractions', steps_per_epoch * (epoch + 1), epoch + 1)
        # Episode return: average, std, min, max
        log_writer.add_scalar('EpisodesReturn/mean', episode_return_mean, epoch + 1)
        # Episode len: average, std, min, max
        log_writer.add_scalar('EpisodeLength/mean', episode_len_mean, epoch + 1)
        # Total loss:
        log_writer.add_scalar('Loss/TotalLoss/Mean', mean_loss_total, epoch + 1)
        log_writer.add_scalar('Loss/ActionLoss/Mean', mean_loss_action, epoch + 1)
        log_writer.add_scalar('Loss/ValueLoss/Mean', mean_loss_value, epoch + 1)
        # Elapsed time
        log_writer.add_scalar('Progress/ElapsedTimeMinutes', (time.time() - start_time) / 60, epoch + 1)
        # Entropy of action outputs
        log_writer.add_scalar('Entropy/mean', mean_entropy, epoch + 1)
        # Approx kl
        log_writer.add_scalar('ApproxKL', approx_kl, epoch + 1)

    log_writer.close()


if __name__ == '__main__':
    import argparse
    import json
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--logdir', type=str)
    args = parser.parse_args()

    with open(args.parameters) as f:
        parameters = json.load(f)

    # Create the directories for logs and saved models
    dir = os.path.dirname(args.logdir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.dirname(args.logdir + 'saved_models/')
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.dirname(args.logdir + 'log/')
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Copy all parameters to log dir
    with open(args.logdir + 'gymParameters.json', 'w') as f:
        json.dump(parameters, f, indent='\t')


    env = gym.make('CartPole-v0')

    from risenet.cartpole_agent import CartpolePolicy
    # from risenet.gymAgent import GymResNetPolicy
    from gym import spaces

    space_dict = {}
    # space_dict['rgb'] = spaces.Box(low=0, high=255, shape=(256, 256, 3))
    space_dict['pointgoal_with_gps_compass'] = env.observation_space
    observation_space = spaces.Dict(space_dict)

    ac = CartpolePolicy(observation_space, env.action_space, hidden_size=32)
    # ac = GymResNetPolicy(observation_space, env.action_space)

    n_hidden_l = ac.net.num_recurrent_layers
    hidden_size = ac.net.output_size

    PPO_trainer(env, ac, num_rec_layers=n_hidden_l, hidden_state_size=hidden_size, seed=parameters['training']['seed'],
                steps_per_epoch=parameters['training']['steps_per_epoch'], epochs=parameters['training']['epochs'],
                minibatch_size=parameters['training']['minibatch_size'],
                gamma=parameters['training']['gamma'], clip_ratio=parameters['training']['clip_ratio'],
                lr=parameters['training']['lr'], train_iters=parameters['training']['train_iters'],
                lam=parameters['training']['lambda'], max_episode_len=parameters['training']['max_episode_len'],
                value_loss_coef=parameters['training']['value_loss_coef'],
                entropy_coef=parameters['training']['entropy_coef'], target_kl=parameters['training']['target_kl'],
                save_freq=parameters['training']['save_freq'], log_dir=args.logdir)
