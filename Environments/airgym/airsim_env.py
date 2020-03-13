import airsim
import gym
from gym import spaces
import numpy as np

from . import agent_controller as ac
from . import utils


# function to align with gym framework
def make(**env_kwargs):
    return AirsimEnv(**env_kwargs)


class AirsimEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 sensors=['depth', 'pointgoal_with_gps_compass'],
                 max_dist=10,
                 height=256,
                 width=256,
                 reward_success=10,
                 reward_failure=-0.5,
                 reward_collision=-10,
                 reward_move_towards_goal=0.01,
                 reward_rotate=-0.01,
                 distance_threshold=0.5,
                 floor_z=0.5,
                 ceiling_z=-1,
                 ):

        self.sensors = sensors
        self.max_dist = max_dist
        self.height = height
        self.width = width
        self.distance_threshold = distance_threshold

        # TODO: add floor and ceiling to parameters
        self.floor_z = floor_z
        self.ceiling_z = ceiling_z

        self.REWARD_SUCCESS = reward_success
        self.REWARD_FAILURE = reward_failure
        self.REWARD_COLLISION = reward_collision
        self.REWARD_MOVE_TOWARDS_GOAL = reward_move_towards_goal
        self.REWARD_ROTATE = reward_rotate

        space_dict = {}
        if 'rgb' in sensors:
            space_dict.update({"rgb": spaces.Box(low=0, high=255, shape=(height, width, 3))})
        if 'depth' in sensors:
            space_dict.update({"depth": spaces.Box(low=0, high=255, shape=(height, width, 1))})
        if 'pointgoal_with_gps_compass' in sensors:
            space_dict.update({"pointgoal_with_gps_compass": spaces.Box(low=0, high=20, shape=(2,))})

        self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.Discrete(6)

        self.agent_dead = True
        self.target_position = None
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def _get_state(self):
        sensor_types = [x for x in self.sensors if x != 'pointgoal_with_gps_compass']
        # get camera images
        observations = utils.get_camera_observation(self.client, sensor_types=sensor_types, max_dist=self.max_dist,
                                                    height=self.height, width=self.width)

        if 'pointgoal_with_gps_compass' in self.sensors:
            compass = utils.get_compass_reading(self.client, self.target_position)
            observations.update({'pointgoal_with_gps_compass': compass})

        return observations

    def reset(self, sub_t=True):
        utils.reset(self.client, sub_t)
        self.agent_dead = False
        self.target_position = utils.generate_target(self.client, self.max_dist / 2, sub_t=True)
        return self._get_state()

    def step(self, action):
        if self.agent_dead:
            print("Episode over. Reset the environment to try again.")
            return None, None, None, {}

        old_distance_to_target = self._get_state()['pointgoal_with_gps_compass'][0]
        reward = 0
        info = {'env':"AirSim"}
        # actions: [terminate, move forward, rotate left, rotate right, ascend, descend]
        if action == 0:
            success = utils.target_found(self.client, self.target_position, threshold=self.distance_threshold)
            if success:
                reward += self.REWARD_SUCCESS
                self.client.simPrintLogMessage(
                    "SUCCESS - Terminated at target. Position: {}".format(utils.get_position(self.client)))
                info['terminated_at_target'] = True
            else:
                reward += self.REWARD_FAILURE
                self.client.simPrintLogMessage(
                    "FAILURE - Terminated not at target. Position: {}".format(utils.get_position(self.client)))
                info['terminated_at_target'] = False

            self.target_position = utils.generate_target(self.client, self.max_dist / 2, sub_t=True)
        elif action == 1:
            ac.move_forward(self.client)
        elif action == 2:
            ac.rotate_left(self.client)
            reward += self.REWARD_ROTATE
        elif action == 3:
            ac.rotate_right(self.client)
            reward += self.REWARD_ROTATE
        elif action == 4:
            ac.move_up(self.client)
            reward += self.REWARD_ROTATE
        elif action == 5:
            ac.move_down(self.client)
            reward += self.REWARD_ROTATE

        episode_over = utils.has_collided(self.client, floor_z=self.floor_z, ceiling_z=self.ceiling_z)
        if episode_over:
            self.agent_dead = True
            reward += self.REWARD_COLLISION

        observation = self._get_state()
        # movement reward
        new_distance_to_target = observation['pointgoal_with_gps_compass'][0]
        movement = old_distance_to_target - new_distance_to_target
        movement_threshold = 0.05/self.max_dist
        if action != 0 and action != 2 and action != 3:
            if movement > movement_threshold:
                reward += self.REWARD_MOVE_TOWARDS_GOAL
            elif movement < -movement_threshold:
                reward -= self.REWARD_MOVE_TOWARDS_GOAL
        self.client.simPrintLogMessage("Goal distance, direction ",
                                       str([observation['pointgoal_with_gps_compass'][0],
                                            observation['pointgoal_with_gps_compass'][1]*180/3.14]))
        self.client.simPrintLogMessage("Step reward:", str(reward))
        return observation, reward, episode_over, info

    def get_obstacles(self, field_of_view, n_gridpoints=8):
        assert 'depth' in self.sensors  # make sure depth camera is used
        observations = utils.get_camera_observation(self.client, sensor_types=['depth'], max_dist=self.max_dist,
                                                    height=self.height, width=self.width)
        depth = observations['depth']

        h_idx = np.linspace(0, self.height, n_gridpoints, endpoint=False, dtype=np.int)
        w_idx = np.linspace(0, self.width, n_gridpoints, endpoint=False, dtype=np.int)

        center_x = self.width // 2
        center_y = self.height // 2
        focal_len = self.width / (2 * np.tan(field_of_view / 2))

        points = []
        for v in h_idx:
            for u in w_idx:
                x = depth[v, u]
                if x < self.max_dist:
                    y = (u - center_x) * x / focal_len
                    z = (v - center_y) * x / focal_len
                    points.append([x, y, z])
        point_cloud = np.array(points)

        # Get position and orientation of client
        pos_vec = self.client.simGetGroundTruthKinematics().position
        client_pos = np.array([[pos_vec.x_val, pos_vec.y_val, pos_vec.z_val]])
        client_orientation = self.client.simGetGroundTruthKinematics().orientation

        pitch, roll, yaw = airsim.to_eularian_angles(client_orientation)
        pitch, roll, yaw = -pitch, -roll, -yaw

        rot_mat = np.array([
            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll),
             np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll),
             np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
                            ])

        # get global_points, i.e. compensate for client orientation and position
        global_points = rot_mat @ point_cloud.transpose() + client_pos.transpose()
        global_points = global_points.transpose()

        return global_points

    def render(self, mode='human'):
        pass

    def close(self):
        # self.airsim_process.terminate()     # TODO: does not work
        pass
