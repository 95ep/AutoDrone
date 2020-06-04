# Autonomous Drone

This repository contains the work that was done for the Master's Thesis: *Autonomous Mapping of Unknown Environments Using a UAV*.
Deep reinforcement learning was used to find an autonomous drone behavior that explores scenes in *Airsim* UAV simulator.

The repository contains the following modules:
* Environments: 2 custom reinfocement learning environments, and a helper class that can be used to process most Open AI Gym environments.
    * Airgym: Airsim RL environment.
    * Exploration: A map utility that also works as an RL environment for an exploration task. Extends Airgym to combine the exploration task with Airsim.  
* NeuralNetwork: creates an actor-critic type neural network agent, based on the specified RL environment.
* Parameters: different parameter files suitable for different environments.
* main script and proximal policy trainer.

## Proximal Policy Optimization

*PPO_trainer.py* and *PPO_utils.py* can be used to easily train a neural network agent on different RL environments.
Experience is collected by letting the agent interact with the environment and the data is stored in a buffer class.
After collecting enough interactions for an epoch, a training loss is calculated in the style of PPO using the data in
the buffer and the parameters of the neural network agent are updated using stochastic gradient descent.

*manual_experience_collection.py* can be used to collect data by controlling the UAV using the keyboard, potentially
speeding up the training process by providing data that better shows an efficient policy.

## Main

The main script sets up everything that is needed to train or evaluate an agent in some RL environment that is supported
by the *env_utils.py* file in the *Environments* module.
The main script is executed with one or two arguments. If training is to be performed, the script is called together with
a parameter file describing the specific settings and a directory path which will store log files and saved checkpoints.

    python main.py --parameters ./Parameters/parameters_pong.json --logdir ./Runs/Pong_experiment_0/

To test or evaluate an existing neural network agent, the *mode* field in the parameter file must be change from *training*
to *evaluation* and the field *eval/weights* is the agent being evaluated. In that case of evaluation, the log directory path is not necessary.

    python main.py --parameters ./Parameters/parameters_pong.json

## Full system and detection verification
The verify_detection script can be used to verify how object and obstacle detection performs in an Airsim scene.
The script is set up to use a trained local navigator to visit hard coded waypoints in order to explore the Viktoria scene.
If used in another scene the waypoints and ground truth positions of the objects must be adjusted.
After visiting all waypoints four numpy arrays are saved to the --save_dir folder. These arrays are the positions detected
objects and obstacles, the ground truth positions of the objects and the positions visited by the UAV during the flight.

    python verify_detection.py --parameters ./Parameters/parameters_verify_objs.json --save_dir ./runs/verification/

The full system can be evaluated using the _full_system.py_ file. This file puts together the trained local navigator
agent with a global planner agent or random walk depending on the parameter file. Four metrics are calculated and stored
in the --logdir: # detected objects, precision, recall and # cells explored as a function of steps.

This script imports information about the ground truth positions of the objects from the _verify_detection.py_ file,
which right now are the values for the Viktoria office. If another AirSim map is used remember to update accordingly.

    python full_system.py --parameters ./Parameters/parameters_full.json --logdir ./runs/full_system_eval/

## Install
(Python 3)

Pytorch is required to run and train the neural networks, installations differ depending on system. The same is true for
OpenAI Gym with Atari environments. The addition of Atari environments is not necessary unless they are used to showcase
the DRL agent. In order to simulate the UAV the airsim python package is needed. More information on how to install airsim
can be seen here https://github.com/microsoft/AirSim .

The following Python version and packages were used in the project:
* Python 3.7.6
* AirSim 1.2.5
* Pytorch 1.4.0+cu92
* Numpy 1.18.1
* OpenAI Gym 0.15.4
* vtkplotter 2020.2.2
* scikit-learn 0.22.2
* OpenCV-Contrib-Python 3.4.2.16
