import numpy as np
import torch
import torch.nn as nn

"""
class VanillaNet(Policy):

    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()  # super(NeuralNetwork, self).__init_()

        self.output_dim = output_dim[0]

        self.input_dim = input_dim[0]
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, self.output_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, input):

        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        policies[t,:] = policy.squeeze()
        values[t] = value.squeeze()

        return value, action,

"""

# ======================================================================
# ======================================================================
# ======================================================================
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import CategoricalNet, Flatten


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def replace_output_layer(self, dim_actions):
        self.action_distribution = CategoricalNet(
            self.net.output_size, dim_actions
        )

    def replace_prev_action_embedding_layer(self, dim_actions):
        self.net.prev_action_embedding = nn.Embedding(dim_actions, 32)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states



class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward lass

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def single_forward(self, x, hidden_states, masks):
        r"""Forward for a non-sequence input
        """
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def seq_forward(self, x, hidden_states, masks):
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        masks = masks.view(t, n)

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)



class CartpolePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal_with_gps_compass",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
    ):
        super().__init__(
            CartpoleNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
            ),
            action_space.n,
        )


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class CartpoleNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n, 32)
        self._n_prev_action = 32

        self._n_input_goal = (
            observation_space.spaces[self.goal_sensor_uuid].shape[0]
        )
        self.tgt_embeding = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action

        self.fake_rnn = nn.Linear(rnn_input_size, hidden_size)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return True

    @property
    def num_recurrent_layers(self):
        return 1

    def get_tgt_encoding(self, observations):
        goal_observations = observations[self.goal_sensor_uuid]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                goal_observations[:, 1],
                goal_observations[:, 2],
                goal_observations[:, 3]
            ],
            -1,
        )
        return self.tgt_embeding(goal_observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        tgt_encoding = self.get_tgt_encoding(observations)
        prev_actions = self.prev_action_embedding(
            prev_actions.long().squeeze(-1)
        )

        x += [tgt_encoding, prev_actions]
        x = torch.cat(x, dim=1)

        x = F.relu(self.fake_rnn(x))

        return x, (None, None)