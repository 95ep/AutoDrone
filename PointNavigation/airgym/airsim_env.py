import airsim
import gym
from gym import spaces
import subprocess
import json

from . import utils
from . import agent_controller as ac

# with open('./PointNavigation/parameters.json') as f:
#     parameters = json.load(f)
#
# REWARD_SUCCESS = parameters["environment"]['REWARD_SUCCESS']
# REWARD_FAILURE = parameters["environment"]['REWARD_FAILURE']
# REWARD_COLLISION = parameters["environment"]['REWARD_COLLISION']
# REWARD_MOVE_TOWARDS_GOAL = parameters["environment"]['REWARD_MOVE_TOWARDS_GOAL']

REWARD_SUCCESS = 0
REWARD_FAILURE = 0
REWARD_COLLISION = 0
REWARD_MOVE_TOWARDS_GOAL = 0
REWARD_ROTATE = 0


def make(**kwargs):
    global REWARD_SUCCESS
    global REWARD_FAILURE
    global REWARD_COLLISION
    global REWARD_MOVE_TOWARDS_GOAL
    global REWARD_ROTATE
    REWARD_SUCCESS = kwargs['REWARD_SUCCESS']
    REWARD_FAILURE = kwargs['REWARD_FAILURE']
    REWARD_COLLISION = kwargs['REWARD_COLLISION']
    REWARD_MOVE_TOWARDS_GOAL = kwargs['REWARD_MOVE_TOWARDS_GOAL']
    REWARD_ROTATE = kwargs['REWARD_ROTATE']

    return AirsimEnv(kwargs['sensors'], kwargs['max_dist'], kwargs['height'], kwargs['width'])


class AirsimEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, sensors=['depth', 'pointgoal_with_gps_compass'], max_dist=10, height=256, width=256):
        self.sensors = sensors
        self.max_dist = max_dist
        self.height = height
        self.width = width
        self.distance_threshold = 0.5
        self.agent_dead = True
        space_dict = {}
        if 'rgb' in sensors:
            space_dict.update({"rgb": spaces.Box(low=0, high=255, shape=(height, width, 3))})
        if 'depth' in sensors:
            space_dict.update({"depth": spaces.Box(low=0, high=255, shape=(height, width, 1))})
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
        observations = utils.get_camera_observation(self.client, sensor_types=sensor_types, max_dist=self.max_dist, height=self.height, width=self.width)

        if 'pointgoal_with_gps_compass' in self.sensors:
            compass = utils.get_compass_reading(self.client, self.target_position)
            observations.update({'pointgoal_with_gps_compass': compass})

        return observations


    def step(self, action):
        if self.agent_dead:
            print("Episode over. Reset the environment to try again.")
            return None, None, None, (None, None)

        old_distance_to_target = self._get_state()['pointgoal_with_gps_compass'][0]
        reward = 0
        info = {}
        # actions: [terminate, move forward, rotate left, rotate right, ascend, descend, no-op?]
        if action == 0:
            success = utils.target_found(self.client, self.target_position, threshold=self.distance_threshold)
            if success:
                reward += REWARD_SUCCESS
                self.client.simPrintLogMessage("Terminated at target - SUCCESS")
                info['terminated_at_target'] = True
            else:
                reward += REWARD_FAILURE
                self.client.simPrintLogMessage("Terminated not close to target - FAILURE")
                info['terminated_at_target'] = False

            self.target_position = utils.generate_target(self.client, self.max_dist/2, sub_t=True)
        elif action == 1:
            ac.move_forward(self.client)
        elif action == 2:
            ac.rotate_left(self.client)
            reward += REWARD_ROTATE
        elif action == 3:
            ac.rotate_right(self.client)
            reward += REWARD_ROTATE
        elif action == 4:
            ac.move_up(self.client)
        elif action == 5:
            ac.move_down(self.client)
        elif action == 6:
            pass

        episode_over = utils.has_collided(self.client, floor_z=0.5, ceiling_z=-1.0)
        if episode_over:
            self.agent_dead = True
            reward += REWARD_COLLISION

        observation = self._get_state()
        position = utils.get_position(self.client)
        orientation = utils.get_orientation(self.client)
        # reward moving towards the goal
        new_distance_to_target = observation['pointgoal_with_gps_compass'][0]
        movement = old_distance_to_target - new_distance_to_target
        movement_threshold = 0.05/self.max_dist
        if action != 0 and action != 2 and action != 3:
            if movement > movement_threshold:
                reward += REWARD_MOVE_TOWARDS_GOAL
            elif movement < -movement_threshold:
                reward -= REWARD_MOVE_TOWARDS_GOAL
        self.client.simPrintLogMessage("Goal distance, direction ", str([observation['pointgoal_with_gps_compass'][0],
                                                                            observation['pointgoal_with_gps_compass'][1]*180/3.14]))
        self.client.simPrintLogMessage("Step reward:", str(reward))
        return observation, reward, episode_over, info


    def get_obstacles(self, FOV, n_gridpoints=8):
        assert 'depth' in self.sensors # Make sure we're in environment where depth camera is used
        carmera_dict = get_camera_observation(client, sensor_types=['depth'], max_dist=self.max_dist, height=self.height, width=self.width)
        depth = carmera_dict['depth']

        h_idx = np.linspace(0, self.height, n_gridpoints, endpoint=False, dtype=np.int)
        w_idx = np.linspace(0, self.width, n_gridpoints, endpoint=False, dtype=np.int)

        centerX = self.width // 2
        centerY = self.height // 2
        focal_len = width / (2 * np.tan(FOV / 2))

        points = []
        for v in h_idx:
            for u in w_idx:
                X = depth[v,u]
                if X < self.max_dist:
                    Y = (u - centerX) * X / focal_len
                    Z = (v - centerY) * X / focal_len
                    points.append([X, Y, Z])
        point_cloud = np.array(points)

        # Get position and orientation of client
        pos_vec = client.simGetGroundTruthKinematics().position
        client_pos = np.array([[pos_vec.x_val, pos_vec.y_val, pos_vec.z_val]])
        client_orientation = client.simGetGroundTruthKinematics().orientation

        pitch, roll, yaw = airsim.to_eularian_angles(client_orientation)
        pitch, roll, yaw = -pitch, -roll, -yaw

        rot_mat = np.array([
                            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
                            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
                            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
                            ])

        # Get global_points, i.e. compensate for client orientation and position
        global_points = rot_mat @ cleanPoints.transpose() + client_pos.transpose()
        global_points = global_points.transpose()

        return global_points


    def reset(self):
        utils.reset(self.client)
        self.agent_dead = False
        self.target_position = utils.generate_target(self.client, self.max_dist/2, sub_t=True)
        return self._get_state()

    def render(self, mode='human'):
        pass

    def close(self):
        self.airsim_process.terminate()     # TODO: does not work
