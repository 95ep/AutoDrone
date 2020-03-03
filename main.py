import argparse
import json
import os
import torch

from Environments.env_utils import make
from Agents.neutral_net import NeutralNet
from PPO_trainer import PPO_trainer, evaluate


# Create parser and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str)
parser.add_argument('--logdir', type=str, default=None)
args = parser.parse_args()

# Open the json parameters file
with open(args.parameters) as f:
    parameters = json.load(f)

# Create env and env_utils
env_utils, env = make(**parameters)

# Get network kwargs from env_utils
network_kwargs = env_utils.get_network_kwargs()
# Add additional kwargs from parameter file
network_kwargs.update(parameters['neural_network'])

ac = NeutralNet(**network_kwargs)

if parameters['mode'] == 'training':
    # Create the directories for logs and saved models
    parent_dir = os.path.dirname(args.logdir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    models_dir = os.path.dirname(args.logdir + 'saved_models/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    log_dir = os.path.dirname(args.logdir + 'log/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Copy all parameters to log dir
    with open(args.logdir + 'parameters.json', 'w') as f:
        json.dump(parameters, f, indent='\t')

    if parameters['training']['resume_training']:
        ac.load_state_dict(torch.load(parameters['training']['weights']))

    # Start training
    PPO_trainer(env, ac, env_utils, parameters, args['logdir'])

elif parameters['mode'] == 'evaluation':
    log_dict = evaluate(env, env_utils, ac, **parameters['eval'])
    print("Evaluation done. Dict with log values is printed below:")
    print(log_dict)

else:
    print("{} is not a recognized mode.".format(parameters['mode']))
