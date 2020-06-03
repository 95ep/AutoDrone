import argparse, json, torch, os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from Environments.env_utils import make_env_utils
from NeuralNetwork.neural_net import NeuralNet
from verify_detection import gt_monitor_positions, precision_recall


# Create parser and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str)
parser.add_argument('--logdir', type=str, default=None)
args = parser.parse_args()

# Open the json parameters file
with open(args.parameters) as f:
    parameters = json.load(f)

parent_dir = os.path.dirname(args.logdir)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# Copy all parameters to log dir
with open(args.logdir + 'parameters.json', 'w') as f:
    json.dump(parameters, f, indent='\t')

log_writer = SummaryWriter(log_dir=args.logdir + 'log/')

# Create env and env_utils
env_utils, env = make_env_utils(**parameters)

if not parameters['random_exploration']:
    # Get network kwargs from env_utils
    network_kwargs = env_utils.get_network_kwargs()
    # Add additional kwargs from parameter file
    network_kwargs.update(parameters['neural_network'])

    old_shape = network_kwargs['visual_input_shape']
    network_kwargs['visual_input_shape'] = (old_shape[0], old_shape[1], 6)

    ac = NeuralNet(**network_kwargs)

    ac.load_state_dict(torch.load(parameters['weights']))
else:
    mean = np.array([0, 0])
    cov = parameters['sigma'] * np.eye(2)

obs = env.reset()
obs = obs[:,:,0:6]
obs_vector, obs_visual = env_utils.process_obs(obs)
comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
for step in range(parameters['n_steps']):
    if parameters['random_exploration']:
        action = np.random.multivariate_normal(mean, cov)
        action = torch.tensor(action, dtype=torch.float32)
    else:
        with torch.no_grad():
            value, action, log_prob = ac.act(comb_obs)

    next_obs, reward, done, info = env.step(env_utils.process_action(action))
    next_obs = next_obs[:,:,0:6]
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

    if len(object_positions) > 0:
        precision, recall = precision_recall(object_positions, gt_monitor_positions, parameters['Exploration']['cell_scale'])
    else:
        precision = 0
        recall = 0
    n_detected_objects = len(gt_monitor_positions) * recall

    log_writer.add_scalar('n_detected_objects', n_detected_objects, step)
    log_writer.add_scalar('precision', precision, step)
    log_writer.add_scalar('recall', recall, step)
    log_writer.add_scalar('explored', info['explored'], step)
