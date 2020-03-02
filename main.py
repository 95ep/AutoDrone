import argparse
import json

from environments.env_utils import make


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
