# Neural Network

Neural network class that can initiate neural network agents suitable for reinforcement learning problems. 
An actor-critic architecture is used and different encoders and actions heads can be specified to fit many environments 
that follow the Gym framework.

## Details

Three observations types are supported:
* vectors (1D array)
* images (3D array)
* tokens (int)

The neural network is compatible with both discrete and continuous action spaces.

## Example Use 
Initiate actor-critic agent suitable for _Pong_ (visual encoder, categorical action head), obtain observation from
environment, predict action using neural network, update environment and render: 

    import gym
    import torch
    from NeuralNetwork.neural_net import NeuralNet
    
    env = gym.make('Pong-v0')
    actor_critic = NeuralNet(has_vector_encoder=False, 
                             vector_input_shape=None,
                             has_visual_encoder=True, 
                             visual_input_shape=env.observation_space.shape, 
                             channel_config=[16, 32, 64], 
                             kernel_size_config=[3, 3, 3], 
                             padding_config=[1, 1, 1], 
                             max_pool_config=[True, False, False],
                             action_dim=env.action_space.n, 
                             continuous_actions=False, 
                             has_previous_action_encoder=False,
                             hidden_size=32, 
                             num_hidden_layers=2):
                             
     obs = env.reset()
     obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
     obs = (obs,)
     value, action, log_prob = actor_critic.act(obs)
     env.step(action.item())
     env.render()
