import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PPO_utils import ExperienceDataset, ExperienceSampler
import time
import numpy as np
import pickle
import random
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=50, fill='=', print_end="\r"):
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


def _update(actor_critic, data, optimizer, minibatch_size, train_iters,
            target_kl, clip_ratio, value_loss_coef, entropy_coef):
    global approx_kl_iter
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
            (
                total_loss,
                action_loss,
                value_loss,
                entropy,
                approx_kl
            ) = _total_loss(
                minibatch,
                actor_critic,
                clip_ratio,
                value_loss_coef,
                entropy_coef
            )

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


def evaluate(env, env_utils, actor_critic, n_evals=1, n_eval_steps=200):
    log_dict = {'Eval/TotalReturn': 0, 'Eval/nDones': 0, 'Eval/TotalSteps': 0}

    for _ in range(n_evals):
        # Set up interactions with env
        obs_vector, obs_visual = env_utils.process_obs(env.reset())
        comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
        for step in range(n_eval_steps):
            log_dict['Eval/TotalSteps'] += 1
            with torch.no_grad():
                value, action, _ = actor_critic.act(comb_obs, deterministic=True)

            next_obs, reward, done, info = env.step(env_utils.process_action(action))
            env.render()

            if 'env' in info and info['env'] == "AirSim":
                if 'Eval/nTerminationsCorrect' not in log_dict:
                    log_dict['Eval/nTerminationsCorrect'] = 0
                    log_dict['Eval/nTerminationsIncorrect'] = 0
                if 'terminated_at_target' in info:
                    if info['terminated_at_target']:
                        log_dict['Eval/nTerminationsCorrect'] += 1
                    else:
                        log_dict['Eval/nTerminationsIncorrect'] += 1

            log_dict['Eval/TotalReturn'] += reward

            obs_vector, obs_visual = env_utils.process_obs(next_obs)
            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
            if done:
                log_dict['Eval/nDones'] += 1
                break
    if 'env' in info and info['env'] == "Exploration":
        env.render(local=False)  # TODO: ONLY IF EXPLORATION
        import time
        time.sleep(3)
    log_dict['Eval/AvgReturn'] = log_dict['Eval/TotalReturn'] / n_evals
    log_dict['Eval/AvgEpisodeLen'] = log_dict['Eval/TotalSteps'] / n_evals
    return log_dict


def PPO_trainer(env, actor_critic, env_utils, log_dir,
                gamma = 0.99,
                n_epochs = 500,
                steps_per_epoch = 1024,
                minibatch_size = 64,
                clip_ratio = 0.2,
                lr = 1e-4,
                train_iters = 8,
                lam = 0.95,
                max_episode_len = 200,
                value_loss_coef = 0.5,
                entropy_coef = 0.01,
                target_kl = 0.03,
                save_freq = 20,
                resume_training = False,
                epoch_to_resume = 0,
                prob_train_on_manual_experience = 0,
                manual_experience_path = '',
                **eval_kwargs
                ):

    # Set up logger
    log_writer = SummaryWriter(log_dir=log_dir + 'log/')

    # Set up buffer
    buffer = env_utils.make_buffer(steps_per_epoch, gamma, lam)

    # Set up optimizer
    optimizer = Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())), lr=lr)

    # Start timer
    start_time = time.time()

    if resume_training:
        start_epoch = epoch_to_resume
    else:
        start_epoch = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(start_epoch, n_epochs):
        print("Epoch {} started".format(epoch))
        log_dict = {}
        if (epoch % save_freq == 0) or (epoch == n_epochs - 1):
            log_dict = evaluate(env, env_utils, actor_critic, **eval_kwargs)
            torch.save(actor_critic.state_dict(), log_dir + 'saved_models/model{}.pth'.format(epoch))

        if manual_experience_path and np.random.rand() < prob_train_on_manual_experience:
            file_name = random.choice(os.listdir(manual_experience_path))
            with open(manual_experience_path + file_name, 'rb') as f:
                data = pickle.load(f)
                act = data['act']
                obs_vector = data['obs_vector']
                obs_visual = data['obs_visual']
                comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
                with torch.no_grad():
                    _, logp, _ = actor_critic.evaluate_actions(comb_obs, act)
                data['logp'] = logp

        else:
            obs_vector, obs_visual = env_utils.process_obs(env.reset())
            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

            episode_len = 0

            log_dict['Episode/nDones'] = 0
            log_dict['Episode/TotalReturn'] = 0
            log_dict['Episode/TotalSteps'] = 0
            for t in range(steps_per_epoch):
                episode_len += 1
                log_dict['Episode/TotalSteps'] += 1
                with torch.no_grad():
                    value, action, log_prob = actor_critic.act(comb_obs)

                next_obs, reward, done, info = env.step(env_utils.process_action(action))

                if 'env' in info and info['env'] == "AirSim":
                    if 'Episode/nTerminationsCorrect' not in log_dict:
                        log_dict['Episode/nTerminationsCorrect'] = 0
                        log_dict['Episode/nTerminationsIncorrect'] = 0
                    if 'terminated_at_target' in info:
                        if info['terminated_at_target']:
                            log_dict['Episode/nTerminationsCorrect'] += 1
                        else:
                            log_dict['Episode/nTerminationsIncorrect'] += 1

                log_dict['Episode/TotalReturn'] += reward

                buffer.store(obs_vector, obs_visual, action, reward, value, log_prob)

                # Update obs
                obs_vector, obs_visual = env_utils.process_obs(next_obs)
                comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

                timeout = episode_len == max_episode_len
                terminal = done or timeout
                epoch_ended = t == steps_per_epoch - 1
                if done:
                    log_dict['Episode/nDones'] += 1

                if terminal or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        with torch.no_grad():
                            value = actor_critic.get_value(comb_obs)
                    else:
                        value = 0
                    buffer.finish_path(value)

                    # Reset if episode ended
                    obs, episode_len = env.reset(), 0
                    obs_vector, obs_visual = env_utils.process_obs(obs)
                    comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

            log_dict['Episode/AvgReturn'] = log_dict['Episode/TotalReturn'] / (log_dict['Episode/nDones'] + 1)
            log_dict['Episode/AvgLen'] = log_dict['Episode/TotalSteps'] / (log_dict['Episode/nDones'] + 1)
            # A epoch of experience is collected
            data = buffer.get()

        # Perform PPO update
        actor_critic = actor_critic.to(device=device)
        (
            mean_loss_total,
            mean_loss_action,
            mean_loss_value,
            mean_entropy,
            approx_kl
        ) = _update(
            actor_critic,
            data,
            optimizer,
            minibatch_size,
            train_iters,
            target_kl,
            clip_ratio,
            value_loss_coef,
            entropy_coef
        )

        actor_critic = actor_critic.cpu()
        # Calc metrics and log info about epoch
        log_dict['Progress/TotalEnvInteractions'] = steps_per_epoch * (epoch + 1)
        log_dict['Loss/TotalLoss/Mean'] = mean_loss_total
        log_dict['Loss/ActionLoss/Mean'] = mean_loss_action
        log_dict['Loss/ValueLoss/Mean'] = mean_loss_value
        log_dict['Progress/ElapsedTimeMinutes'] = (time.time() - start_time) / 60
        log_dict['Entropy/mean'] = mean_entropy
        log_dict['ApproxKL'] = approx_kl

        # Add everything from log_dict to log_writer
        for k, v in log_dict.items():
            log_writer.add_scalar(k, v, epoch)

    log_writer.close()
    env.close()
