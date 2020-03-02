import argparse
import json

from environments.env_utils import make
from PointNavigation.risenet.neutral_net import NeutralNet


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
