# Environments
This module contains the RL environments used in this project. The main file used is _env_utils.py_ that contains 
functionality for creating the different environments together with a utils_object used for aligning input/output
between the environment and the NN agent.


## airgym
Contains the implementation of the AirSim RL environment.

## Exploration
Contains the implementation of the RL environments for exploring both AirSim scenes and synthetic 2D maps.

## env_utils.py
This file contains the function _make_env_utils(**param_kwargs)_, the superclass EnvUtilsSuper and subclasses for
 each environment available.
 
 
 used to obtain the environment object and
a utilities object. The purpose of the utilities object is to aligning input/output between the environment and the NN agent.

### make_env_utils(**param_kwargs)
This function is used to create the environment object and a utilities object based on the parameters in a parameters
file. The purpose of the utilities object is to aligning input/output between the environment and the NN agent.

The available environments are: CartPole, Pong, AirSim, Exploration, AutonomousDrone.

### class EnvUtilsSuper
This class defines the utilities object class. This class is extended by subclasses for each available environment.
 The purpose of the utilities object is to aligning input/output between the environment and the NN agent. The utils 
 object is also used for creating a experience buffer with the correct dimensions.
 
For more information about these classes and the implemented functions please check out the docstrings.

### Example Use
These lines shows how to first create an environment, agent and buffer and then take one step and add the experience 
to the buffer. Examples of parameter files can be found in the Parameters folder.

The example use is based on _main.py_ in the root folder so please check that script to get an understanding of the
complete PPO chain.
```
import json
from Environments.env_utils import make_env_utils
from NeuralNetwork.neural_net import NeuralNet

# Open the json parameters file
with open('path_to_parameter_file') as f:
    parameters = json.load(f)

# Create env and env_utils
env_utils, env = make_env_utils(**parameters)

# Get network kwargs from env_utils
network_kwargs = env_utils.get_network_kwargs()
# Add additional kwargs from parameter file
network_kwargs.update(parameters['neural_network'])

# Create the actor-critic network
ac = NeuralNet(**network_kwargs)

# Set up buffer
buffer = env_utils.make_buffer(steps_per_epoch=1000, gamma=0.99, lam=0.95)

# Reset env and process observation
observation = env.reset()
obs_vector, obs_visual = env_utils.process_obs(observation)
comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)

# Predict action
with torch.no_grad():
    value, action, log_prob = actor_critic.act(comb_obs)

# Process action and take step in env
processed_action = env_utils.process_action(action)
next_obs, reward, done, info = env.step(processed_action)

# Store one step of experience
buffer.store(obs_vector, obs_visual, action, reward, value, log_prob)

```