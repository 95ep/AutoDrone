import airsim
import gym
from gym import spaces
import time

from . import utils
from . import agent_controller as ac


def make(**kwargs):
    return AirsimEnv(**kwargs)


class AirsimEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, sensors=['depth', 'pointgoal_with_gps_compass'], max_dist=10):
        self.sensors = sensors
        self.max_dist = max_dist
        space_dict = {}
        if 'rgb' in sensors:
            space_dict.update({"rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3))})
        if 'depth' in sensors:
            space_dict.update({"depth": spaces.Box(low=0, high=255, shape=(256, 256, 1))})
        if 'pointgoal_with_gps_compass' in sensors:
            space_dict.update({"pointgoal_with_gps_compass" : spaces.Box(low=0, high=20, shape=(2,))})
        self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.Discrete(6)

        # TODO: start engine automatically
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()
        time.sleep(1)

        # obtain goal
        self.target_position = utils.generate_target(self.client)


    def _get_state(self):
        sensor_types = [x for x in self.sensors if x!='pointgoal_with_gps_compass']
        # get camera images
        observations = utils.get_camera_observation(self.client, sensor_types=sensor_types, max_dist=10)

        if 'pointgoal_with_gps_compass' in self.sensors:
            compass = utils.get_compass_reading(self.client, self.target_position)
            observations.update({'pointgoal_with_gps_compass': compass})

        return observations


    def step(self, action):
        # actions: [terminate, move forward, rotate left, rotate right, ascend, descend, no-op?]
        if action == 0:
            ac.terminate(self.client)
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

        observation = self._get_state()
        reward = 0
        episode_over = False
        return observation, reward, episode_over, {}


    def reset(self):
        return self._get_state()


    def render(self, mode='human'):
        pass


    def close(self):
        pass
