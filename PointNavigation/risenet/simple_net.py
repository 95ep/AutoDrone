import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SimpleNet(nn.Module):
    def __init__(self, input_shape, action_shape, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_shape)
        self.critic = nn.Linear(hidden_size, 1)

    def act(self, obs):
        assert obs.shape[0] == 1
        value, policy = self(obs)
        action = Categorical(policy).sample()
        log_prob = torch.log(policy)
        return value, action, log_prob[action]

    def get_value(self, obs):
        value, policy = self(obs)
        return value

    def evaluate_actions(self, obs, action):
        value, policy = self(obs)
        entropy = Categorical(policy).entropy()
        log_prob = torch.log(policy)
        log_list = [log_prob[i, action[i]].unsqueeze(0) for i in range(len(action))]
        log_prob = torch.cat(log_list)
        log_prob = log_prob.view(len(action), 1)
        return value, log_prob, entropy

    def forward(self, input):
        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return value, policy
