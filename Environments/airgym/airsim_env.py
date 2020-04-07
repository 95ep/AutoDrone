import airsim
import gym
from gym import spaces
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth

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
                 rgb_height=256,
                 rgb_width=256,
                 depth_height=64,
                 depth_width=64,
                 field_of_view=np.pi/2,
                 reward_success=10,
                 reward_failure=-0.5,
                 reward_collision=-10,
                 reward_move_towards_goal=0.01,
                 reward_rotate=-0.01,
                 distance_threshold=0.5,
                 invalid_prob=0.0,
                 floor_z=100,
                 ceiling_z=-100,
                 scene_string=""
                 ):

        self.sensors = sensors
        self.max_dist = max_dist
        self.rgb_height = rgb_height
        self.rgb_width = rgb_width
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.field_of_view = field_of_view
        self.distance_threshold = distance_threshold
        self.invalid_prob = invalid_prob

        self.floor_z = floor_z
        self.ceiling_z = ceiling_z

        self.REWARD_SUCCESS = reward_success
        self.REWARD_FAILURE = reward_failure
        self.REWARD_COLLISION = reward_collision
        self.REWARD_MOVE_TOWARDS_GOAL = reward_move_towards_goal
        self.REWARD_ROTATE = reward_rotate

        if scene_string == "":
            self.scene = None
        else:
            self.scene = scene_string

        space_dict = {}
        if 'rgb' in sensors:
            space_dict.update({"rgb": spaces.Box(low=0, high=255, shape=(rgb_height, rgb_width, 3))})
        if 'depth' in sensors:
            space_dict.update({"depth": spaces.Box(low=0, high=255, shape=(depth_height, depth_width, 1))})
        if 'pointgoal_with_gps_compass' in sensors:
            space_dict.update({"pointgoal_with_gps_compass": spaces.Box(low=0, high=20, shape=(2,))})

        self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.Discrete(6)

        self.collision_t_stamp = None
        self.agent_dead = True
        self.target_position = None
        self.valid_trgt = None
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Add fields for object detection
        self.kp_q_list = []
        self.desc_q_list = []
        self.corners_list = []
        self.min_match_count = 0
        self.rejection_factor = 0

    def _get_state(self):
        sensor_types = [x for x in self.sensors if x != 'pointgoal_with_gps_compass']
        # get camera images
        observations = utils.get_camera_observation(self.client, sensor_types=sensor_types, max_dist=self.max_dist,
                                                    height=self.depth_height, width=self.depth_width)

        if 'pointgoal_with_gps_compass' in self.sensors:
            compass = utils.get_compass_reading(self.client, self.target_position)
            observations.update({'pointgoal_with_gps_compass': compass})

        return observations

    def reset(self, target_position=None):
        utils.reset(self.client, scene=self.scene)
        self.agent_dead = False
        self.collision_t_stamp = None
        if target_position is None:
            self.target_position, self.valid_trgt = utils.generate_target(self.client, self.max_dist / 2,
                                                                          scene=self.scene, invalid_prob=self.invalid_prob)
        else:
            self.target_position = target_position
        return self._get_state()

    def step(self, action):
        if self.agent_dead:
            print("Episode over. Reset the environment to try again.")
            return None, None, None, {}

        old_distance_to_target = self._get_state()['pointgoal_with_gps_compass'][0]
        reward = 0
        info = {'env': "AirSim"}
        # actions: [terminate, move forward, rotate left, rotate right, ascend, descend]
        if action == 0:
            success = utils.target_found(self.client, self.target_position, self.valid_trgt, threshold=self.distance_threshold)
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

            self.target_position, self.valid_trgt = utils.generate_target(self.client, self.max_dist / 2,
                                                                          scene=self.scene, invalid_prob=self.invalid_prob)
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

        episode_over = self.has_collided()
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

    def has_collided(self):
        new_collision_t_stamp = self.client.simGetCollisionInfo().time_stamp
        if self.collision_t_stamp is None:
            self.collision_t_stamp = new_collision_t_stamp
        if new_collision_t_stamp != self.collision_t_stamp:
            self.collision_t_stamp = new_collision_t_stamp
            self.client.simPrintLogMessage("Collision with object")
            return True
        else:
            z_pos = self.get_position()[2]
            if z_pos > self.floor_z:
                self.client.simPrintLogMessage("Collision with floor")
                return True
            elif z_pos < self.ceiling_z:
                self.client.simPrintLogMessage("Collision with ceiling")
                return True
            else:
                return False

    def get_position(self):
        position = utils.get_position(self.client)
        return np.array([position.x_val, position.y_val, position.z_val])

    def get_orientation(self):
        orientation = utils.get_orientation(self.client)
        return np.array([np.cos(orientation), np.sin(orientation)])

    def get_obstacles(self, field_of_view, n_gridpoints=8):
        assert 'depth' in self.sensors  # make sure depth camera is used
        observations = utils.get_camera_observation(self.client, sensor_types=['depth'], max_dist=self.max_dist,
                                                    height=self.depth_height, width=self.depth_width)
        depth = observations['depth']
        depth = depth.squeeze()
        h_idx = np.linspace(0, self.depth_height, n_gridpoints, endpoint=False, dtype=np.int)
        w_idx = np.linspace(0, self.depth_width, n_gridpoints, endpoint=False, dtype=np.int)
        u_grid, v_grid = np.meshgrid(w_idx, h_idx)
        points_2d = [(u, v) for u, v in zip(u_grid.flatten(), v_grid.flatten())]

        point_cloud = utils.reproject_2d_points(points_2d, depth, self.max_dist, field_of_view)
        if point_cloud.shape[0] > 0:
            obstacles_3d_coords = utils.local2global(point_cloud, self.client)
        else:
            obstacles_3d_coords = np.array([], dtype=float)

        return obstacles_3d_coords

    def setup_object_detection(self, query_paths, rejection_factor, min_match_thres):
        assert len(self.kp_q_list) == 0

        self.rejection_factor = rejection_factor
        self.min_match_count = min_match_thres
        for path in query_paths:
            query_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            h, w = query_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            sift = cv.xfeatures2d.SIFT_create()
            kp_q, desc_q = sift.detectAndCompute(query_image, None)
            self.kp_q_list.append(kp_q)
            self.desc_q_list.append(desc_q)
            self.corners_list.append(pts)

    def get_trgt_objects(self):
        assert len(self.kp_q_list) > 0
        observations = utils.get_camera_observation(self.client, sensor_types=['rgb'], max_dist=self.max_dist,
                                                    height=self.rgb_height, width=self.rgb_width)
        train_image = observations['rgb'].astype(np.uint8)
        train_image = cv.cvtColor(train_image, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create()
        kp_t, desc_t = sift.detectAndCompute(train_image, None)
        if len(kp_t) == 0:
            global_points = np.array([], dtype=float)
            return global_points
        x = np.array([kp_t[i].pt for i in range(len(kp_t))])
        bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
        if bandwidth < 0.1:
            # Not possible to form clusters
            return np.array([], dtype=float)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms.fit(x)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)

        kp_per_cluster = []
        desc_per_cluster = []
        for i in range(n_clusters):
            d, = np.where(labels == i)
            kp_per_cluster.append([kp_t[xx] for xx in d])
            desc_per_cluster.append([desc_t[xx] for xx in d])

        homographies = []
        dst_list = []

        for i in range(n_clusters):
            kp_cluster = kp_per_cluster[i]
            desc_cluster = np.array(desc_per_cluster[i], dtype=np.float32)

            # Brute force matching
            # BFMatcher with default params
            bf = cv.BFMatcher()
            good_list = []
            for descQ in self.desc_q_list:
                matches = bf.knnMatch(descQ, desc_cluster, k=2)
                # Apply ratio test
                good = []
                if len(kp_cluster) > 1:
                    for m, n in matches:
                        if m.distance < self.rejection_factor * n.distance:
                            good.append(m)
                good_list.append(good)

            # Find homography
            max_inliers = 0
            homography_tmp = None
            dst_tmp = None
            for j, good in enumerate(good_list):

                pts = self.corners_list[j]
                if len(good) > self.min_match_count:
                    src_pts = np.float32([self.kp_q_list[j][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_cluster[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    if homography is not None:
                        matches_mask = mask.ravel().tolist()
                        dst = cv.perspectiveTransform(pts, homography)
                        non_zero_masks = np.nonzero(matches_mask)

                        no_outliers = True
                        for idx in non_zero_masks[0]:
                            if int(cv.pointPolygonTest(np.array(dst),
                                                       tuple(dst_pts[idx, 0, :].astype(np.int)), False)) \
                                    != 1:
                                no_outliers = False
                                break

                        if no_outliers and len(non_zero_masks[0]) > max_inliers:
                            max_inliers = len(non_zero_masks[0])
                            homography_tmp = homography
                            dst_tmp = dst

            if max_inliers > 0:
                dst_list.append(dst_tmp)
                homographies.append(homography_tmp)
        obj_2d_coords = []
        for dst in dst_list:
            # Calculate center point
            x = np.sum(dst[:, 0, 0]) / 4
            y = np.sum(dst[:, 0, 1]) / 4
            # Rescale to match depth img dimension
            w_scale = self.depth_width / self.rgb_width
            h_scale = self.depth_height / self.rgb_height
            x = int(x * w_scale)
            y = int(y * h_scale)

            if x > 0 and x < self.depth_width and y > 0 and y < self.depth_height:
                obj_2d_coords.append((x, y))

        if len(obj_2d_coords) > 0:
            observations = utils.get_camera_observation(self.client, sensor_types=['depth'], max_dist=self.max_dist,
                                                        height=self.depth_height, width=self.depth_width)
            depth = observations['depth'].squeeze()
            point_cloud = utils.reproject_2d_points(obj_2d_coords, depth, self.max_dist, self.field_of_view)
            if point_cloud.shape[0] > 0:
                global_points = utils.local2global(point_cloud, self.client)
            else:
                global_points = np.array([], dtype=float)
        else:
            global_points = np.array([], dtype=float)
        return global_points

    def render(self, mode='human'):
        pass

    def close(self):
        # self.airsim_process.terminate()     # TODO: does not work
        pass
