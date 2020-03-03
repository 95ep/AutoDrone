import numpy as np
import torch
import gym
from gym.wrappers.frame_stack import FrameStack
from cv2 import resize, INTER_CUBIC
import json

import PointNavigation.airgym as airgym  # TODO - is this correct syntax for import?
import Exploration.exploration_dev as exploration_dev


def make(**kwargs):
    if kwargs['training']['env_str'] == 'CartPole':
        env_utils_obj = EnvUtilsCartPole()
        env = env_utils_obj.make_env()
    elif kwargs['training']['env_str'] == 'Atari':
        env_utils_obj = EnvUtilsAtari(**kwargs)
        env = env_utils_obj.make_env()
    elif kwargs['training']['env_str'] == 'AirSim':
        env_utils_obj = EnvUtilsAirSim(**kwargs)
        env = env_utils_obj.make_env()
    elif kwargs['training']['env_str'] == 'Exploration':
        env_utils_obj = EnvUtilsExploration()
        env = env_utils_obj.make_env()
    else:
        raise ValueError("env_str not recognized.")

    assert env_utils_obj is not None and env is not None

    return env_utils_obj, env


class EnvUtilsSuper:
    def __init__(self):
        self.network_kwargs = {'has_previous_action_encoder': False, 'has_vector_encoder': False,
                               'vector_input_shape': tuple(), 'has_visual_encoder': False,
                               'visual_input_shape': tuple(), 'action_dim': None}

    def make_env(self, **kwargs):
        raise NotImplementedError

    def get_network_kwargs(self):
        return self.network_kwargs

    def process_obs(self, obs_from_env):
        raise NotImplementedError

    def process_action(self, action):
        return action.item()


class EnvUtilsCartPole(EnvUtilsSuper):
    def __init__(self):
        super().__init__()

    def make_env(self):
        env = gym.make('CartPole-v0')
        self.network_kwargs['has_vector_encoder'] = True
        self.network_kwargs['vector_input_shape'] = env.observation_space.shape
        self.network_kwargs['action_dim'] = env.action_space.n
        return env

    def process_obs(self, obs_from_env):
        obs_visual = None
        obs_vector = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual


class EnvUtilsAtari(EnvUtilsSuper):
    def __init__(self, parameters):
        self.frame_stack = parameters['training']['frame_stack']
        self.height = parameters['training']['height']
        self.width = parameters['training']['width']
        super().__init__()

    def make_env(self):
        env = gym.make('PongDeterministic-v4')
        env = FrameStack(env, self.frame_stack)
        self.network_kwargs['has_visual_encoder'] = True
        self.network_kwargs['visual_input_shape'] = (self.height, self.width, 3 * self.frame_stack)
        self.network_kwargs['action_dim'] = env.action_space.n
        return env

    def process_obs(self, obs_from_env):
        obs_vector = None
        # To np array and put in range (0,1)
        ary = np.array(obs_from_env.__array__(), dtype=np.float32) / 255
        ary = np.concatenate(ary, axis=-1)
        c = ary.shape[2]
        new_ary = np.zeros((self.height, self.width, c), dtype=np.float32)
        for i in range(c):
            new_ary[:, :, i] = resize(ary[:, :, i], dsize=(self.height, self.width), interpolation=INTER_CUBIC)
        obs_visual = torch.clamp(torch.as_tensor(new_ary).unsqueeze(0), 0, 1)

        return obs_vector, obs_visual


class EnvUtilsAirSim(EnvUtilsSuper):
    def __init__(self, parameters):
        self.max_dist = parameters['environment']['max_dist']
        self.height = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
        self.width = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
        super().__init__()

    def make_env(self, **parameters):
        # Write AirSim settings to a json file
        with open(parameters['training']['airsim_settings_path'], 'w') as f:
            json.dump(parameters['airsim'], f, indent='\t')
        input(
            'Copied AirSim settings to Documents folder. \n (Re)Start AirSim and then press enter to start training...')

        env = airgym.make(**parameters)
        if 'rgb' in parameters['environment']['sensors']:
            self.network_kwargs['has_visual_encoder'] = True
            self.network_kwargs['visual_input_shape'] = (self.height, self.width, 3)

        if 'depth' in parameters['environment']['sensors']:
            if self.network_kwargs['has_visual_encoder']:
                visual_shape = self.network_kwargs['visual_input_shape']
                self.network_kwargs['visual_input_shape'] = (visual_shape[0], visual_shape[1], visual_shape[2]+1)
            else:
                self.network_kwargs['has_visual_encoder'] = True
                self.network_kwargs['visual_input_shape'] = (self.height, self.width, 1)

        if 'pointgoal_with_gps_compass' in parameters['environment']['sensors']:
            self.network_kwargs['has_vector_encoder'] = True
            self.network_kwargs['vector_input_shape'] = (3,)

        self.network_kwargs['action_dim'] = env.action_space.n

        return env

    def process_obs(self, obs_from_env):
        obs_vector = None
        obs_visual = None

        if 'pointgoal_with_gps_compass' in obs_from_env:
            dist = obs_from_env['pointgoal_with_gps_compass'][0]
            dist = np.clip(dist, None, self.max_dist) / self.max_dist
            angle = obs_from_env['pointgoal_with_gps_compass'][1]
            obs_vector = torch.as_tensor([dist, np.sin(angle), np.cos(angle)], dtype=torch.float32).unsqueeze(0)

        if 'rgb' in obs_from_env and 'depth' in obs_from_env:
            rgb = torch.as_tensor(obs_from_env['rgb'], dtype=torch.float32) / 255.0
            depth = np.clip(obs_from_env['depth'], 0, self.max_dist) / self.max_dist
            depth = torch.as_tensor(depth, dtype=torch.float32)
            obs_visual = torch.cat((rgb, depth), dim=2).unsqueeze(0)

        elif 'rgb' in obs_from_env:
            obs_visual = torch.as_tensor(obs_from_env['rgb'], dtype=torch.float32).unsqueeze(0) / 255

        elif 'depth' in obs_from_env:
            depth = np.clip(obs_from_env['depth'], 0, self.max_dist) / self.max_dist
            obs_visual = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual


class EnvUtilsExploration(EnvUtilsSuper):
    def __init__(self):
        super().__init__()

    def make_env(self):
        env = exploration_dev.make()
        self.network_kwargs['has_visual_encoder'] = True
        self.network_kwargs['visual_input_shape'] = env.observation_space.shape
        self.network_kwargs['action_dim'] = env.action_space.shape[0]
        return env

    def process_obs(self, obs_from_env):
        obs_vector = None
        obs_visual = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual

    def process_action(self, action):
        return action.squeeze().numpy()
