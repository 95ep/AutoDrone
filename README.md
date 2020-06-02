# Autonomous Drone 

This repository contains the work that was done for the Master's Thesis: *Autonomous Mapping of Unknown Environments Using a UAV*.
Deep reinforcement learning was used to find an autonomous drone behavior that explores scenes in *Airsim* UAV simulator.

The repository contains the following modules:
* Environments: 2 custom reinfocement learning environments, and a helper class that can be used to process most Open AI Gym environments.
    * Airgym: Airsim RL environment.
    * Exploration: A map utility that also works as an RL environment for an exploration task. Extends Airgym to combine the exploration task with Airsim.  
* NeuralNetwork: creates an actor-critic type neural network agent, based on the specified RL environment.
* ObjectDetection: feature extraction based object detection using SIFT.
* Parameters: different parameter files suitable for different environments.
* main script and proximal policy trainer.

## Proximal Policy Optimization

*PPO_trainer.py* and *PPO_utils.py* can be used to easily train a neural network agent on different RL environments. 
Experience is collected by letting the agent interact with the environment and the data is stored in a buffer class.
After collecting enough interactions for an epoch, a training loss is calculated in the style of PPO using the data in 
the buffer and the parameters of the neural network agent are updated using stochastic gradient descent. 

## Main

The main script sets up everything that is needed to train or evaluate an agent in some RL environment that is supported 
by the *env_utils.py* file in the *Environments* module. 
The main script is executed with one or two arguments. If training is to be performed, the script is called together with
a parameter file describing the specific settings and a directory path which will store log files and saved checkpoints. 

    python main.py --parameters ./Parameters/parameters_pong.json --logdir ./Runs/Pong_experiment_0/

To test or evaluate an existing neural network agent, the *mode* field in the parameter file must be change from *training* 
to *evaluation*. In that case, the log directory path is not necessary.

    python main.py --parameters ./Parameters/parameters_pong.json

## Install
(Python 3)

Pytorch is required to run and train the neural networks, installations differ depending on system. The same is true for 
OpenAI Gym with Atari environments. The addition of Atari environments is not necessary unless they are used to showcase 
the DRL agent. In order to simulate the UAV the airsim python package is needed. More information on how to install airsim
can be seen here https://github.com/microsoft/AirSim .

Other non-standard modules that are required:
* vtkplotter
* sklearn
* OpenCV
