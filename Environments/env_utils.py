import numpy as np
import torch
import gym
from gym.wrappers.frame_stack import FrameStack
from cv2 import resize, INTER_CUBIC
import json
import copy

from PPO_utils import PPOBuffer


def make_env_utils(**param_kwargs):
    """
    Function that creates the RL environment as well as a utility object that is used to
    process observation and actions in order to make them compatible with between the NN agent
    and the env.
    :param param_kwargs: parameters dict describing which env to create and the properties of the env
    :return: env_utils_obj and env
    """
    env_str = param_kwargs['env_str']
    if env_str == 'CartPole':
        env_utils_obj = EnvUtilsCartPole()
        env = env_utils_obj.make_env()
    elif env_str == 'Pong':
        env_utils_obj = EnvUtilsPong(**param_kwargs[env_str])
        env = env_utils_obj.make_env()
    elif env_str == 'AirSim':
        env_utils_obj = EnvUtilsAirSim(**param_kwargs[env_str])
        env = env_utils_obj.make_env()
    elif env_str == 'Exploration':
        env_utils_obj = EnvUtilsExploration(**param_kwargs[env_str])
        env = env_utils_obj.make_env()
    elif env_str == 'AutonomousDrone':
        env_utils_obj = EnvUtilsAutonomousDrone(**param_kwargs['Exploration'], **param_kwargs)
        env = env_utils_obj.make_env()
    else:
        raise ValueError("env_str not recognized.")

    assert env_utils_obj is not None and env is not None

    return env_utils_obj, env


class EnvUtilsSuper:
    """
    The superclass that the other EnvUtils classes extends. Defines a number of functions, some of which
    are overridden by one or multiple subclasses.
    """
    def __init__(self):
        """
        Constructor of superclass. Sets a few default values for the network_kwargs dict.
        """
        self.network_kwargs = {'has_previous_action_encoder': False, 'has_vector_encoder': False,
                               'vector_input_shape': tuple(), 'has_visual_encoder': False,
                               'visual_input_shape': tuple(), 'action_dim': None}

    def make_env(self):
        """
        Function that creates the env object. Not implemented in superclass and overridden by all subclasses.
        :return: N/A
        """
        raise NotImplementedError

    def make_buffer(self, steps_per_epoch, gamma, lam):
        """
        Creates the buffer used to store experience in the PPO algorithm
        :param steps_per_epoch: (int) number of steps per epoch
        :param gamma: Discount factor for return
        :param lam: lambda factor used in Generalized Advantage Estimation.
        :return: The PPO buffer object
        """
        if self.network_kwargs['has_vector_encoder']:
            vector_shape = self.network_kwargs['vector_input_shape']
        else:
            vector_shape = None
        if self.network_kwargs['has_visual_encoder']:
            visual_shape = self.network_kwargs['visual_input_shape']
        else:
            visual_shape = None

        return PPOBuffer(steps_per_epoch, vector_shape, visual_shape, 1, gamma, lam)

    def get_network_kwargs(self):
        """
        Returns the network kwargs of the NN agent connected to the environment.
        :return:
        """
        return self.network_kwargs

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. Are None if the type of observation is not used in the environment.
         Not implemented in superclass and overridden by all subclasses.
        :param obs_from_env: The observation to process.
        :return: N/A
        """
        raise NotImplementedError

    def process_action(self, action):
        """
        Processes the action outputed by the NN agent to match the action space of the env.
        :param action: The action to process.
        :return: Action matching expected format of env
        """
        return action.item()


class EnvUtilsCartPole(EnvUtilsSuper):
    """
    Subclass for the CartPole env extending the EnvUtilsSuper class.
    """
    def __init__(self):
        super().__init__()

    def make_env(self):
        """
        Creates the CartPole env and updates the network kwargs related to the envs observation and action spaces.
        :return: the CartPole env object
        """
        env = gym.make('CartPole-v0')
        self.network_kwargs['has_vector_encoder'] = True
        self.network_kwargs['vector_input_shape'] = env.observation_space.shape
        self.network_kwargs['action_dim'] = env.action_space.n
        return env

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. obs_visual is None for this env.
        :param obs_from_env: The observation to process.
        :return: obs_vector and obs_visual
        """
        obs_visual = None
        obs_vector = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual


class EnvUtilsPong(EnvUtilsSuper):
    """
    Subclass for the CartPole env extending the EnvUtilsSuper class.
    """
    def __init__(self, **pong_kwargs):
        """
        Constructor for EnvUtilsPong class. Sets attributes defining shape of observation.
        :param pong_kwargs: Kwargs dict defining shape of observation
        """
        self.frame_stack = pong_kwargs['frame_stack']
        self.height = pong_kwargs['height']
        self.width = pong_kwargs['width']
        super().__init__()

    def make_env(self):
        """
        Creates the Pong env and updates the network kwargs related to the envs observation and action spaces.
        :return: the Pong env object.
        """
        env = gym.make('PongDeterministic-v4')
        env = FrameStack(env, self.frame_stack)
        self.network_kwargs['has_visual_encoder'] = True
        self.network_kwargs['visual_input_shape'] = (self.height, self.width, 3 * self.frame_stack)
        self.network_kwargs['action_dim'] = env.action_space.n
        return env

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. obs_vector is None for this env.
        :param obs_from_env: The observation to process.
        :return: obs_vector and obs_visual
        """
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
    """
    Subclass for the AirSim env extending the EnvUtilsSuper class.
    """
    def __init__(self, **airsim_kwargs):
        """
        Constructor for EnvUtilsAirSim class. Sets attributes defining shape of observation and other aspects of
        the environment
        :param airsim_kwargs: Kwargs dict defining the environment
        """
        self.airgym_kwargs = copy.copy(airsim_kwargs['airgym_kwargs'])
        self.rgb_height = airsim_kwargs['airsim_settings']['CameraDefaults']['CaptureSettings'][0]['Height']
        self.rgb_width = airsim_kwargs['airsim_settings']['CameraDefaults']['CaptureSettings'][0]['Width']
        self.depth_height = airsim_kwargs['airsim_settings']['CameraDefaults']['CaptureSettings'][1]['Height']
        self.depth_width = airsim_kwargs['airsim_settings']['CameraDefaults']['CaptureSettings'][1]['Width']
        self.airgym_kwargs['rgb_height'] = self.rgb_height
        self.airgym_kwargs['rgb_width'] = self.rgb_width
        self.airgym_kwargs['depth_height'] = self.depth_height
        self.airgym_kwargs['depth_width'] = self.depth_width
        self.airgym_kwargs['field_of_view'] = \
            airsim_kwargs['airsim_settings']['CameraDefaults']['CaptureSettings'][0]['FOV_Degrees'] / 180 * np.pi
        self.max_dist = self.airgym_kwargs['max_dist']

        self.settings_path = airsim_kwargs['airsim_settings_path']
        self.airsim_settings = airsim_kwargs['airsim_settings']

        super().__init__()

    def make_env(self):
        """
        Creates the AirSim env and updates the network kwargs related to the envs observation and action spaces.
        :return: the AirSim env object
        """
        import Environments.airgym as airgym

        # Write AirSim settings to a json file
        with open(self.settings_path, 'w') as f:
            json.dump(self.airsim_settings, f, indent='\t')
        input(
            'Copied AirSim settings to Documents folder. \n (Re)Start AirSim and then press enter to start training...')

        env = airgym.make(**self.airgym_kwargs)
        if 'rgb' in self.airgym_kwargs['sensors']:
            self.network_kwargs['has_visual_encoder'] = True
            self.network_kwargs['visual_input_shape'] = (self.rgb_height, self.rgb_width, 3)

        if 'depth' in self.airgym_kwargs['sensors']:
            if self.network_kwargs['has_visual_encoder']:
                assert self.rgb_height == self.depth_height, \
                    "rgb_height is {} and depth_height".format(self.rgb_height, self.depth_height)
                assert self.rgb_width == self.depth_width, \
                    "rgb_width is {} and depth_width".format(self.rgb_width, self.depth_width)
                visual_shape = self.network_kwargs['visual_input_shape']
                self.network_kwargs['visual_input_shape'] = (visual_shape[0], visual_shape[1], visual_shape[2] + 1)
            else:
                self.network_kwargs['has_visual_encoder'] = True
                self.network_kwargs['visual_input_shape'] = (self.depth_height, self.depth_width, 1)

        if 'pointgoal_with_gps_compass' in self.airgym_kwargs['sensors']:
            self.network_kwargs['has_vector_encoder'] = True
            self.network_kwargs['vector_input_shape'] = (3,)

        self.network_kwargs['action_dim'] = env.action_space.n

        return env

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. obs_vector and obs_visual are None if not used.
        :param obs_from_env: The observation to process.
        :return: obs_vector and obs_visual
        """
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
    """
    Subclass for the Exploration env extending the EnvUtilsSuper class.
    """
    def __init__(self, **exploration_kwargs):
        super().__init__()
        self.exploration_kwargs = exploration_kwargs

    def make_env(self):
        """
        Creates the Exploration env and updates the network kwargs related to the envs observation and action spaces.
        :return: the Exploration env object
        """
        import Environments.Exploration.map_env as map_env
        env = map_env.make(**self.exploration_kwargs)
        self.network_kwargs['has_visual_encoder'] = True
        self.network_kwargs['continuous_actions'] = True
        self.network_kwargs['visual_input_shape'] = env.observation_space.shape
        self.network_kwargs['action_dim'] = env.action_space.shape[0]
        return env

    def make_buffer(self, steps_per_epoch, gamma, lam):
        """
        Creates the buffer used to store experience in the PPO algorithm
        :param steps_per_epoch: (int) number of steps per epoch
        :param gamma: Discount factor for return
        :param lam: lambda factor used in Generalized Advantage Estimation.
        :return: The PPO buffer object
        """
        if self.network_kwargs['has_vector_encoder']:
            vector_shape = self.network_kwargs['vector_input_shape']
        else:
            vector_shape = None
        if self.network_kwargs['has_visual_encoder']:
            visual_shape = self.network_kwargs['visual_input_shape']
        else:
            visual_shape = None

        action_shape = self.network_kwargs['action_dim']

        return PPOBuffer(steps_per_epoch, vector_shape, visual_shape, action_shape, gamma, lam)

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. obs_vector is None for this env.
        :param obs_from_env: The observation to process.
        :return: obs_vector and obs_visual
        """
        obs_vector = None
        obs_visual = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual

    def process_action(self, action):
        """
        Processes the action outputed by the NN agent to match the action space of the env.
        :param action: The action to process.
        :return: Action matching expected format of env
        """
        return action.squeeze().numpy()


class EnvUtilsAutonomousDrone(EnvUtilsSuper):
    """
    Subclass for the AutonomousDrone env extending the EnvUtilsSuper class.
    """
    def __init__(self, **autonomous_drone_kwargs):
        """
        Constructor for EnvUtilsAutonomousDrone class. Sets attributes defining shape of observation and other
        aspects of the environment.
        :param autonomous_drone_kwargs: Kwargs dict defining the environment
        """
        super().__init__()
        self.autonomous_drone_kwargs = autonomous_drone_kwargs

    def make_env(self):
        """
        Creates the AutonomousDrone env and updates the network kwargs related to the envs observation and
        action spaces.
        :return: the AutonomousDrone env object
        """
        import Environments.Exploration.map_env as map_env
        env = map_env.make(**self.autonomous_drone_kwargs)
        self.network_kwargs['has_visual_encoder'] = True
        self.network_kwargs['continuous_actions'] = True
        self.network_kwargs['visual_input_shape'] = env.observation_space.shape
        self.network_kwargs['action_dim'] = env.action_space.shape[0]
        return env

    def make_buffer(self, steps_per_epoch, gamma, lam):
        """
        Creates the buffer used to store experience in the PPO algorithm
        :param steps_per_epoch: (int) number of steps per epoch
        :param gamma: Discount factor for return
        :param lam: lambda factor used in Generalized Advantage Estimation.
        :return: The PPO buffer object
        """
        if self.network_kwargs['has_vector_encoder']:
            vector_shape = self.network_kwargs['vector_input_shape']
        else:
            vector_shape = None
        if self.network_kwargs['has_visual_encoder']:
            visual_shape = self.network_kwargs['visual_input_shape']
        else:
            visual_shape = None

        action_shape = self.network_kwargs['action_dim']

        return PPOBuffer(steps_per_epoch, vector_shape, visual_shape, action_shape, gamma, lam)

    def process_obs(self, obs_from_env):
        """
        Processes the observation from the env and creates an vector obs and an visual obs matching the expected
        input format of the NN agent. obs_vector is None for this env.
        :param obs_from_env: The observation to process.
        :return: obs_vector and obs_visual
        """
        obs_vector = None
        obs_visual = torch.as_tensor(obs_from_env, dtype=torch.float32).unsqueeze(0)

        return obs_vector, obs_visual

    def process_action(self, action):
        """
        Processes the action outputed by the NN agent to match the action space of the env.
        :param action: The action to process.
        :return: Action matching expected format of env
        """
        return action.squeeze().numpy()
