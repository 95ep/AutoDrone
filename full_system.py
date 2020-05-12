import argparse, json, torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from Environments.env_utils import make_env_utils
from Agents.neutral_net import NeutralNet
from verify_obj_detection import gt_monitor_positions, precision_recall

# Create parser and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str)
parser.add_argument('--logdir', type=str, default=None)
args = parser.parse_args()

# Open the json parameters file
with open(args.parameters) as f:
    parameters = json.load(f)

log_writer = SummaryWriter(log_dir=args.logdir + 'log/')

# Create env and env_utils
env_utils, env = make_env_utils(**parameters)

if not parameters['random_exploration']:
    # Get network kwargs from env_utils
    network_kwargs = env_utils.get_network_kwargs()
    # Add additional kwargs from parameter file
    network_kwargs.update(parameters['neural_network'])

    ac = NeutralNet(**network_kwargs)

    ac.load_state_dict(torch.load(parameters['exploration']['weights']))

obs_vector, obs_visual = env_utils.process_obs(env.reset())
comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
for step in range(parameters['n_steps']):
    if parameters['random_exploration']:
        raise NotImplementedError
    else:
        with torch.no_grad():
            value, action, log_prob = ac.act(comb_obs)

    next_obs, reward, done, info = env.step(env_utils.process_action(action))
    obs_vector, obs_visual = env_utils.process_obs(next_obs)
    comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

    # Extract objects found
    map = env._get_map(local=False, binary=False)
    # Extract objects
    obj_map = map == env.tokens['object']
    obj_cells = np.argwhere(obj_map)
    object_positions = []
    for cell in obj_cells:
        object_positions.append(env._get_position(cell))

    precision, recall = precision_recall(object_positions, gt_monitor_positions, parameters['Exploration']['cell_scale'])
    n_detected_objects = len(gt_monitor_positions) * recall

    log_writer.add_scalar('n_detected_objects', n_detected_objects, step)
    log_writer.add_scalar('precision', precision, step)
    log_writer.add_scalar('recall', recall, step)
    log_writer.add_scalar('explored', info['explored'], step)