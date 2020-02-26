import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical




def create_vector_encoder(input_dim, layer_config=[32, 32]):
    num_layers = len(layer_config)
    layer_config = [input_dim] + layer_config
    layer_stack = []
    for i in range(num_layers):
        layer_stack.append(nn.Linear(layer_config[i], layer_config[i+1]))
        layer_stack.append(nn.LeakyReLU())

    output_dim = layer_config[-1]

    return nn.Sequential(*layer_stack), output_dim


def create_visual_encoder(input_shape, channel_config=[16, 32, 64], kernel_size_config=[3, 3, 3],
                          padding_config=[1, 1, 1], max_pool_config=[True, True, False]):
    # TODOs:
    # - calculate output dimension for 'non-same' paddings
    input_channels = input_shape[2]
    num_layers = len(channel_config)
    channel_config = [input_channels] + channel_config
    layer_stack = []
    size_reduction = 1
    for i in range(num_layers):
        temp_layer = nn.Conv2d(channel_config[i],
                               channel_config[i+1],
                               kernel_size=kernel_size_config[i],
                               padding=padding_config[i])
        layer_stack.append(temp_layer)
        layer_stack.append(nn.LeakyReLU())
        if max_pool_config[i]:
            layer_stack.append(nn.MaxPool2d(kernel_size=2, stride=2))
            size_reduction *= 2

    layer_stack.append(nn.Flatten())

    output_dim = input_shape[0] * input_shape[1] * channel_config[-1] / (size_reduction ** 2)
    assert float(output_dim).is_integer()  # should be integer. If not, check that input shape is correct

    return nn.Sequential(*layer_stack), int(output_dim)


def create_compass_encoder(input_dim, encoding_dim):
    return nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.LeakyReLU()), int(encoding_dim)


def create_previous_action_encoder(action_dim, encoding_dim):
    # +1 for 'null' action, when there is no previous action
    return nn.Embedding(action_dim + 1, encoding_dim), int(encoding_dim)


class CategoricalActor(nn.Module):
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Linear(hidden_size, num_actions)

    def _distribution(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)

    def forward(self, x):
        return self._distribution(x)


class NeutralNet(nn.Module):
    def __init__(self, has_vector_encoder=True, vector_input_shape=(4,),
                 has_visual_encoder=True, visual_input_shape=(128, 128, 2),
                 channel_config=[16, 32, 64], kernel_size_config=[3, 3, 3],
                 padding_config=[1, 1, 1], max_pool_config=[True, True, False],
                 has_compass_encoder=True, compass_input_shape=(3,),
                 num_actions=6, has_previous_action_encoder=False,
                 hidden_size=32, num_hidden_layers=2):

        super().__init__()
        self.has_vector_encoder = has_vector_encoder
        self.has_visual_encoder = has_visual_encoder
        self.has_compass_encoder = has_compass_encoder
        self.has_previous_action_encoder = has_previous_action_encoder

        concatenation_size = 0
        if self.has_vector_encoder:
            self.vector_encoder, output_dim = create_vector_encoder(vector_input_shape[0],
                                                                    layer_config=[hidden_size, hidden_size])
            concatenation_size += output_dim

        if self.has_visual_encoder:
            self.visual_encoder, output_dim = create_visual_encoder(visual_input_shape, channel_config=channel_config,
                                                                    kernel_size_config=kernel_size_config,
                                                                    padding_config=padding_config, max_pool_config=max_pool_config)
            concatenation_size += output_dim

        if self.has_compass_encoder:
            self.compass_encoder, output_dim = create_compass_encoder(compass_input_shape[0], hidden_size)
            concatenation_size += output_dim

        if self.has_previous_action_encoder:
            self.previous_action_encoder, output_dim = create_previous_action_encoder(num_actions, hidden_size)
            concatenation_size += output_dim

        hidden_stack = [nn.Linear(concatenation_size, hidden_size), nn.LeakyReLU()]
        for i in range(1, num_hidden_layers):
            hidden_stack.append(nn.Linear(hidden_size, hidden_size))
            hidden_stack.append(nn.LeakyReLU())

        self.hidden_layers = nn.Sequential(*hidden_stack)

        #self.actor = nn.Linear(hidden_size, num_actions)
        self.actor = CategoricalActor(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def act(self, obs, deterministic=False):
        value, policy_distribution = self(obs)
        if deterministic:
            action = policy_distribution.probs.argmax(dim=-1, keepdim=True)
        else:
            action = policy_distribution.sample()

        log_prob = policy_distribution.log_prob(action)
        return value, action, log_prob

    def get_value(self, obs):
        value, policy = self(obs)
        return value

    def evaluate_actions(self, obs, action):
        value, policy_distribution = self(obs)
        entropy = policy_distribution.entropy().mean()
        log_prob = policy_distribution.log_prob(action.squeeze().long()).unsqueeze(-1)

        return value, log_prob, entropy

    def forward(self, input):
        # inputs should be ordered: vector, visual, compass, prev_action
        # types: float, float, float, long
        idx = 0
        encodings = []
        if self.has_vector_encoder:
            encodings.append(self.vector_encoder(input[idx]))
            idx += 1

        if self.has_visual_encoder:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            visual_input = input[idx].permute(0, 3, 1, 2)
            encodings.append(self.visual_encoder(visual_input))
            idx += 1

        if self.has_compass_encoder:
            encodings.append(self.compass_encoder(input[idx]))
            idx += 1

        if self.has_previous_action_encoder:
            encodings.append(self.previous_action_encoder(input[idx]))
            idx += 1

        x = torch.cat(encodings, dim=1)
        x = self.hidden_layers(x)

        policy_distribution = self.actor(x)
        value = self.critic(x)

        return value, policy_distribution