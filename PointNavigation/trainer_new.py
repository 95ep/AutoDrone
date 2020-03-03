import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dictloader import ExperienceDataset, ExperienceSampler
import time
import numpy as np
import scipy.signal


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_progress_bar (iteration, total, prefix = '', suffix = '',
                        decimals = 1, length = 50, fill = '=', print_end = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str) █
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ' ' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        time.sleep(0.05)
        print('\r ' + ' ' * (filled_length + 10), end=print_end)


def discount_cumsum(x, discount):
    # Helper function. Some magic from scipy.
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def _total_loss(data, actor_critic, clip_ratio, value_loss_coef, entropy_coef):
    act, ret, adv, logp_old = data['act'], data['ret'], data['adv'], data['logp']

    obs_vector = None
    obs_visual = None
    if 'obs_vector' in data:
        obs_vector = data['obs_vector'].to(device=device)
    if 'obs_visual' in data:
        obs_visual = data['obs_visual'].to(device=device)

    comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

    act = act.to(device=device)
    ret = ret.to(device=device)
    adv = adv.to(device=device)
    logp_old = logp_old.to(device=device)

    values, logp, dist_entropy = actor_critic.evaluate_actions(comb_obs, act)
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    action_loss = -torch.min(surr1, surr2).mean()

    value_loss = (ret - values).pow(2).mean()

    total_loss = value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef

    # Compute approx kl for logging purposes
    approx_kl = (logp_old-logp).mean().item()

    return total_loss, action_loss, value_loss, dist_entropy, approx_kl


def _update(actor_critic, buffer, train_iters, optimizer, clip_ratio, value_loss_coef,
            entropy_coef, target_kl, minibatch_size):
    # Spinning up used target_kl for early stopping. Is it smart thing to include?
    global approx_kl_iter
    data = buffer.get()
    total_loss_in_epoch = []
    action_loss_in_epoch = []
    value_loss_in_epoch = []
    entropy_in_epoch = []

    dataset = ExperienceDataset(data)
    sampler = ExperienceSampler(dataset, minibatch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_size=minibatch_size, sampler=sampler)

    for i in range(train_iters):
        print_progress_bar(i+1, train_iters)
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
            print_progress_bar(1, 1)
            print("Early stopping at step {} due to reaching max kl.".format(i))
            break

    mean_total_loss = np.mean(np.array(total_loss_in_epoch))
    mean_action_loss = np.mean(np.array(action_loss_in_epoch))
    mean_value_loss = np.mean(np.array(value_loss_in_epoch))
    mean_entropy = np.mean(np.array(entropy_in_epoch))

    return mean_total_loss, mean_action_loss, mean_value_loss, mean_entropy, np.array(approx_kl_iter).mean()


class PPOBuffer:
    """
    Stores trajectories collected.
    """
    def __init__(self, steps, vector_shape, visual_shape, action_dim, gamma, lam):
        # Constructor - set xxx_shape to None if not used
        if vector_shape is not None:
            self.obs_vector = np.zeros((steps, *vector_shape), dtype=np.float32)
        if visual_shape is not None:
            self.obs_visual = np.zeros((steps, *visual_shape), dtype=np.float32)

        self.act_buf = np.zeros((steps, action_dim), dtype=np.int)
        self.rew_buf = np.zeros(steps, dtype=np.float32)
        self.adv_buf = np.zeros((steps, 1), dtype=np.float32)
        self.ret_buf = np.zeros((steps, 1), dtype=np.float32)
        self.val_buf = np.zeros(steps, dtype=np.float32)
        self.logp_buf = np.zeros((steps, 1), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0  # Keeps track of number of interactions in buffer
        self.path_start_idx = 0
        self.max_size = steps

    def store(self, obs_vector, obs_visual, act, rew, val, logp):
        """
        Add one step of interactions to the buffer, set obs_xx to None (or whatever) if not used
        """
        assert self.ptr < self.max_size
        if hasattr(self, 'obs_vector'):
            self.obs_vector[self.ptr] = obs_vector
        if hasattr(self, 'obs_visual'):
            self.obs_visual[self.ptr] = obs_visual
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
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
        # return all data in the form of torch tensors
        assert self.ptr == self.max_size  # buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        # Advantage normalization
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        if hasattr(self, 'obs_vector'):
            data['obs_vector'] = self.obs_vector
        if hasattr(self, 'obs_visual'):
            data['obs_visual'] = self.obs_visual

        # Convert into torch tensors
        data = {k: torch.as_tensor(v) for k, v in data.items()}

        return data


def PPO_trainer(env, actor_critic, parameters, log_dir):

    # Extract parameters
    env_str = parameters['training']['env_str']
    steps_per_epoch = parameters['training']['steps_per_epoch']
    max_episode_len = parameters['training']['max_episode_len']
    epochs = parameters['training']['epochs']
    minibatch_size = parameters['training']['minibatch_size']
    gamma = parameters['training']['gamma']
    lam = parameters['training']['lambda']
    clip_ratio = parameters['training']['clip_ratio']
    lr = parameters['training']['lr']
    train_iters = parameters['training']['train_iters']
    value_loss_coef = parameters['training']['value_loss_coef']
    entropy_coef = parameters['training']['entropy_coef']
    target_kl = parameters['training']['target_kl']
    save_freq = parameters['training']['save_freq']
    eval_freq = parameters['training']['eval_freq']
    n_eval = parameters['training']['n_eval']

    # Set up logger
    log_writer = SummaryWriter(log_dir=log_dir + 'log/')

    # Get some env variables
    obs_space = env.observation_space

    # Set up experience buffer
    vector_shape = None
    visual_shape = None
    continuous_actions = False
    if env_str == 'CartPole':
        vector_shape = obs_space.shape
    elif env_str == 'Atari':
        visual_shape = (parameters['training']['height'], parameters['training']['width'], 3*parameters['training']['frame_stack'])
    elif env_str == 'AirSim':
        if 'rgb' in parameters['environment']['sensors']:
            h = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
            w = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
            visual_shape = (h, w, 3)

        if 'depth' in parameters['environment']['sensors']:
            h = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
            w = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
            visual_shape = (h, w, 1)

        if 'rgb' in parameters['environment']['sensors'] and 'depth' in parameters['environment']['sensors']:
            h = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
            w = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
            visual_shape = (h, w, 4)

        if 'pointgoal_with_gps_compass' in parameters['environment']['sensors']:
            vector_shape = (3,)
    elif env_str == 'Maze':
        continuous_actions = True
        visual_shape = obs_space.shape
    else:
        raise NameError('env_str not recognized')

    if continuous_actions:
        buffer = PPOBuffer(steps_per_epoch, vector_shape, visual_shape, env.action_space.shape[0], gamma, lam)
    else:
        buffer = PPOBuffer(steps_per_epoch, vector_shape, visual_shape, 1, gamma, lam)
    optimizer = Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())), lr=lr)

    # Prepare for interaction with env
    start_time = time.time()
    obs, episode_return, episode_len = env.reset(), 0, 0
    obs_vector, obs_visual = process_obs(obs, env_str, parameters)
    comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

    if parameters['training']['resume_training']:
        start_epoch = parameters['training']['epoch_to_resume']
    else:
        start_epoch = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(start_epoch, epochs):
        print("Epoch {} started".format(epoch))
        if epoch % eval_freq == 0:
            if env_str == 'AirSim':
                total_ret_eval = 0
                n_collisions_eval = 0
                n_terminate_correct_eval = 0
                n_terminate_incorrect_eval = 0
                done = False
                for step in range(n_eval): # Use n_eval as steps to evaluate in
                    with torch.no_grad():
                        value, action, _ = actor_critic.act(comb_obs, deterministic=True)
                    if continuous_actions:
                        next_obs, reward, done, info = env.step(action.squeeze().numpy())
                    else:
                        next_obs, reward, done, info = env.step(action.item())
                    if 'terminated_at_target' in info:
                        if info['terminated_at_target']:
                            n_terminate_correct_eval += 1
                        else:
                            n_terminate_incorrect_eval += 1
                    total_ret_eval += reward
                    obs_vector, obs_visual = process_obs(next_obs, env_str, parameters)
                    comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
                    if done:
                        obs, episode_return, episode_len = env.reset(), 0, 0
                        obs_vector, obs_visual = process_obs(obs, env_str, parameters)
                        comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
                        n_collisions_eval += 1

                obs, episode_return, episode_len = env.reset(), 0, 0
                obs_vector, obs_visual = process_obs(obs, env_str, parameters)
                comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

                log_writer.add_scalar('Eval/returnTot', total_ret_eval, epoch)
                log_writer.add_scalar('Eval/nCollisions', n_collisions_eval, epoch)
                log_writer.add_scalar('Eval/nTerminationsCorrect', n_terminate_correct_eval, epoch)
                log_writer.add_scalar('Eval/nTerminationsIncorrect', n_terminate_incorrect_eval, epoch)
            else:
                total_eval_ret = 0
                for i in range(n_eval):
                    done = False
                    env.render()
                    while not done:
                        with torch.no_grad():
                            value, action, _ = actor_critic.act(comb_obs, deterministic=True)
                        if continuous_actions:
                            next_obs, reward, done, info = env.step(action.squeeze().numpy())
                        else:
                            next_obs, reward, done, info = env.step(action.item())
                        env.render()
                        total_eval_ret += reward
                        obs_vector, obs_visual = process_obs(next_obs, env_str, parameters)
                        comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
                        if done:
                            obs, episode_return, episode_len = env.reset(), 0, 0
                            obs_vector, obs_visual = process_obs(obs, env_str, parameters)
                            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

                print("Evalutation reward: ", total_eval_ret)
                log_writer.add_scalar('Eval/returnMean', total_eval_ret/n_eval, epoch)

        # list of episode returns and episode lengths to put to logg
        episode_returns_epoch = []
        episode_len_epoch = []
        n_collisions = 0
        n_terminate_correct = 0
        n_terminate_incorrect = 0
        for t in range(steps_per_epoch):
            with torch.no_grad():
                value, action, log_prob = actor_critic.act(comb_obs)

            if continuous_actions:
                next_obs, reward, done, info = env.step(action.squeeze().numpy())
            else:
                next_obs, reward, done, info = env.step(action.item())
            if env_str == 'Airsim':
                if 'terminated_at_target' in info:
                    if info['terminated_at_target']:
                        n_terminate_correct += 1
                    else:
                        n_terminate_incorrect += 1

            episode_return += reward
            episode_len += 1

            buffer.store(obs_vector, obs_visual, action, reward, value, log_prob)

            # Update obs
            obs_vector, obs_visual = process_obs(next_obs, env_str, parameters)
            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

            timeout = episode_len == max_episode_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1
            if done:
                n_collisions += 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    with torch.no_grad():
                        value = actor_critic.get_value(comb_obs)
                else:
                    value = 0
                buffer.finish_path(value)

                episode_returns_epoch.append(episode_return)
                episode_len_epoch.append(episode_len)
                # Reset if episode ended
                obs, episode_return, episode_len = env.reset(), 0, 0
                obs_vector, obs_visual = process_obs(obs, env_str, parameters)
                comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

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
        log_writer.add_scalar('Progress/TotalEnvInteractions', steps_per_epoch * (epoch + 1), epoch)
        log_writer.add_scalar('EpisodesReturn/mean', episode_return_mean, epoch)
        log_writer.add_scalar('EpisodeLength/mean', episode_len_mean, epoch)
        log_writer.add_scalar('Episode/nCollisions', n_collisions, epoch)
        if env_str == 'AirSim':
            log_writer.add_scalar('Episode/nTerminationsCorrect', n_terminate_correct, epoch)
            log_writer.add_scalar('Episode/nTerminationsIncorrect', n_terminate_incorrect, epoch)
        log_writer.add_scalar('Loss/TotalLoss/Mean', mean_loss_total, epoch)
        log_writer.add_scalar('Loss/ActionLoss/Mean', mean_loss_action, epoch)
        log_writer.add_scalar('Loss/ValueLoss/Mean', mean_loss_value, epoch)
        # Elapsed time
        log_writer.add_scalar('Progress/ElapsedTimeMinutes', (time.time() - start_time) / 60, epoch)
        # Entropy of action outputs
        log_writer.add_scalar('Entropy/mean', mean_entropy, epoch)
        # Approx kl
        log_writer.add_scalar('ApproxKL', approx_kl, epoch)

    log_writer.close()
    env.close()
