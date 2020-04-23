import argparse
import json
import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter

from Environments.env_utils import make_env_utils
from Agents.neutral_net import NeutralNet
from PPO_trainer import evaluate


# Create parser and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str)
parser.add_argument('--checkpoints', type=str, default=None)
parser.add_argument('--logdir', type=str)
parser.add_argument('--eval_start', type=int, default=0)
args = parser.parse_args()

# Open the json parameters file
with open(args.parameters) as f:
    parameters = json.load(f)

log_dir = os.path.dirname(args.logdir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Copy all parameters to log dir
with open(log_dir + 'parameters.json', 'w') as f:
    json.dump(parameters, f, indent='\t')

# Create env and env_utils
env_utils, env = make_env_utils(**parameters)

# Get network kwargs from env_utils
network_kwargs = env_utils.get_network_kwargs()
# Add additional kwargs from parameter file
network_kwargs.update(parameters['neural_network'])

ac = NeutralNet(**network_kwargs)


checkpoint_files = [f for f in glob.glob(args.checkpoints + "**/*.pth", recursive=True)]
epoch_and_path_list = []
for f in checkpoint_files:
    _, substring = f.split('model', 1)
    substring, _ = substring.split('.')
    _, substring = substring.split('model')
    epoch = int(substring)
    epoch_and_path_list.append((epoch, f))

epoch_and_path_list.sort(key = lambda x:x[0])

log_writer = SummaryWriter(log_dir=args.logdir)
for epoch, path in epoch_and_path_list:
    if epoch < args.eval_start:
        continue

    ac.load_state_dict(torch.load(path))
    print("Evaluating epoch {}".format(epoch))

    log_dict = evaluate(env, env_utils, ac, **parameters['eval'])
    print("Evaluation done. Dict with log values is printed below:")
    print(log_dict)
    for k, v in log_dict.items():
        log_writer.add_scalar(k, v, epoch)

env.close()
