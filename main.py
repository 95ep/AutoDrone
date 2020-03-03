import argparse
import json
import os
import torch

from environments.env_utils import make
from PointNavigation.risenet.neutral_net import NeutralNet
from PointNavigation.trainer_new import PPO_trainer


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

mode = parameters['foo'] # TODO - Proper extraction
if mode == 'training':
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
    with open(args.logdir + 'parameters.json', 'w') as f:
        json.dump(parameters, f, indent='\t')

    if parameters['training']['resume_training']:
        ac.load_state_dict(torch.load(parameters['training']['weights']))

    # Start training
    PPO_trainer(**foo) # TODO - proper arguments

elif mode == 'eval':
    pass
