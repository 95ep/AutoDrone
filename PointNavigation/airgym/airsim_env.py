import airsim
import gym
from gym import spaces
import subprocess
import json

from . import utils
from . import agent_controller as ac

with open('./PointNavigation/parameters.json') as f:
    parameters = json.load(f)

REWARD_SUCCESS = parameters["environment"]['REWARD_SUCCESS']
REWARD_FAILURE = parameters["environment"]['REWARD_FAILURE']
REWARD_COLLISION = parameters["environment"]['REWARD_COLLISION']
REWARD_MOVE_TOWARDS_GOAL = parameters["environment"]['REWARD_MOVE_TOWARDS_GOAL']


def make(**kwargs):
    return AirsimEnv(**kwargs)


class AirsimEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, sensors=['depth', 'pointgoal_with_gps_compass'], max_dist=10):

        #self.airsim_process = subprocess.Popen(
        #    'C:/Users/Filip/Documents/Skola/Exjobb/Blocks/Blocks.exe')

        self.sensors = sensors
        self.max_dist = max_dist
        self.distance_threshold = 0.5
        self.agent_dead = True
        space_dict = {}
        if 'rgb' in sensors:
            space_dict.update({"rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3))})
        if 'depth' in sensors:
            space_dict.update({"depth": spaces.Box(low=0, high=255, shape=(256, 256, 1))})
        if 'pointgoal_with_gps_compass' in sensors:
            space_dict.update({"pointgoal_with_gps_compass" : spaces.Box(low=0, high=20, shape=(2,))})
        self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.Discrete(6)

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.target_position = None

    def _get_state(self):
        sensor_types = [x for x in self.sensors if x != 'pointgoal_with_gps_compass']
        # get camera images
        observations = utils.get_camera_observation(self.client, sensor_types=sensor_types, max_dist=10)

        if 'pointgoal_with_gps_compass' in self.sensors:
            compass = utils.get_compass_reading(self.client, self.target_position, self.max_dist)
            observations.update({'pointgoal_with_gps_compass': compass})

        return observations

    def step(self, action):
        if self.agent_dead:
            print("Episode over. Reset the environment to try again.")
            return None, None, None, (None, None)

        old_distance_to_target = self._get_state()['pointgoal_with_gps_compass'][0]
        reward = 0
        # actions: [terminate, move forward, rotate left, rotate right, ascend, descend, no-op?]
        if action == 0:
            success = utils.target_found(self.client, self.target_position, self.max_dist, threshold=self.distance_threshold)
            if success:
                reward += REWARD_SUCCESS
            else:
                reward += REWARD_FAILURE
            self.target_position = utils.generate_target(self.client, self.max_dist/4)
        elif action == 1:
            ac.move_forward(self.client)
        elif action == 2:
            ac.rotate_left(self.client)
        elif action == 3:
            ac.rotate_right(self.client)
        elif action == 4:
            ac.move_up(self.client)
        elif action == 5:
            ac.move_down(self.client)
        elif action == 6:
            pass

        episode_over = utils.has_collided(self.client)
        if episode_over:
            self.agent_dead = True
            reward += REWARD_COLLISION

        observation = self._get_state()
        position = utils.get_position(self.client)
        orientation = utils.get_orientation(self.client)
        # reward moving towards the goal
        new_distance_to_target = observation['pointgoal_with_gps_compass'][0]
        movement = old_distance_to_target - new_distance_to_target
        movement_threshold = 0.05    # Give no reward when rotating
        if movement > movement_threshold:
            reward += REWARD_MOVE_TOWARDS_GOAL
        if movement < -movement_threshold:
            reward -= REWARD_MOVE_TOWARDS_GOAL

        return observation, reward, episode_over, (position, orientation)

    def reset(self):
        utils.reset(self.client)
        self.agent_dead = False
        self.target_position = utils.generate_target(self.client, self.max_dist/4)
        return self._get_state()

    def render(self, mode='human'):
        pass

    def close(self):
        self.airsim_process.terminate()     # TODO: does not work
