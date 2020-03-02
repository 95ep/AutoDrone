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
        # TODO - any fields for superclass? Logwriter for instance?
        pass

    def make_env(self, **kwargs):
        raise NotImplementedError

    def process_obs(self, obs_from_env):
        raise NotImplementedError

    def add_log_entries(self):
        raise NotImplementedError


class EnvUtilsCartPole(EnvUtilsSuper):
    def __init__(self):
        super().__init__()

    def make_env(self):
        return gym.make('CartPole-v0')

    def process_obs(self, obs_from_env):
        obs_visual = None
        obs_vector = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual

    def add_log_entries(self):
        # TODO - Implement
        raise NotImplementedError


class EnvUtilsAtari(EnvUtilsSuper):
    def __init__(self, parameters):
        self.frame_stack = parameters['training']['frame_stack']
        self.height = parameters['training']['height']
        self.width = parameters['training']['width']
        super().__init__()

    def make_env(self):
        env = gym.make('PongDeterministic-v4')
        env = FrameStack(env, self.frame_stack)
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

    def add_log_entries(self):
        # TODO - Implement
        raise NotImplementedError


class EnvUtilsAirSim(EnvUtilsSuper):
    def __init__(self, parameters):
        self.max_dist = parameters['environment']['max_dist']
        super().__init__()

    def make_env(self, **parameters):
        # Write AirSim settings to a json file
        with open(parameters['training']['airsim_settings_path'], 'w') as f:
            json.dump(parameters['airsim'], f, indent='\t')
        input(
            'Copied AirSim settings to Documents folder. \n (Re)Start AirSim and then press enter to start training...')

        env = airgym.make(**parameters)

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

    def add_log_entries(self):
        raise NotImplementedError


class EnvUtilsExploration(EnvUtilsSuper):
    def __init__(self):
        super().__init__()

    def make_env(self):
        return exploration_dev.make()

    def process_obs(self, obs_from_env):
        obs_vector = None
        obs_visual = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual

    def add_log_entries(self):
        # TODO - Implement
        raise NotImplementedError
