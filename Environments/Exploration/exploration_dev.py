from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time
sys.path.append('../../Map')
sys.path.append('../Map')
sys.path.append('./Map')

from map_dev import GlobalMap


def make():
    return MapEnv()

class MapEnv:

    def __init__(self, max_steps=64):
        self.cell_map = self.create_map()
        self.direction = 0  # 0 radians = [1,0], pi/2 radians = [0,1]
        self.position = self.cell_map.get_current_position()
        self.reward_scaling = (self.cell_map.vision_range / self.cell_map.cell_scale[0]) * \
                              (self.cell_map.vision_range / self.cell_map.cell_scale[1]) * np.pi
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.cell_map.local_map_size[0], self.cell_map.local_map_size[1],
                                                   len(self.cell_map.cell_map.keys())*self.cell_map.local_map_size[2]),
                                            dtype=np.int)
        self.action_space = spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]),
                                       dtype=np.float32)
        self.steps = 0
        self.max_steps = max_steps
        plt.ion()
        fig = plt.figure()
        self.ax = fig.subplots()

    def create_map(self, map_idx=0):
        """
        Creates a custom environment map representing a room / maze.
        :param map_idx: index representing the different available maps
        :return: a maze environment
        """
        assert map_idx == 0, 'currently there only exists {} different environments'.format(1)
        if map_idx == 0:
            x_size = 61
            y_size = 61
            border_thickness = 10
            z = 0
            starting_positions = [(18, 18, z), (0, 0, z), (-18, -18, z), (-18, 18, z), (18, -18, z),
                                 (0, 15, z), (0, -15, z), (15, 0, z), (-15, 0, z)]
            m = GlobalMap(map_size=(x_size, y_size, 1),
                          cell_scale=(1, 1, 1),
                          starting_position=random.choice(starting_positions),
                          buffer_distance=(0, 0, 0),
                          local_map_size=(11, 11, 1),
                          detection_threshold_obstacle=1,
                          detection_threshold_object=1,
                          fov_angle=3.141592/2,
                          vision_range=4,
                          )
            # make walls
            for x in range(x_size):
                for y in range(0, border_thickness):
                    m._mark_cell(m._get_cell([x-x_size/2, y-y_size/2, z]), 'obstacle')
                    m._mark_cell(m._get_cell([x-x_size/2, y_size/2-y-1, z]), 'obstacle')

            for y in range(y_size):
                for x in range(0, border_thickness):
                    m._mark_cell(m._get_cell([x-x_size/2, y-y_size/2, z]), 'obstacle')
                    m._mark_cell(m._get_cell([x_size/2-x-1, y-y_size/2, z]), 'obstacle')

            # define some obstacles / rooms
            for x in range(4, 7):
                for y in range(-13, -6):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')
            for x in range(4, 13):
                for y in range(-6, -3):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')

            for x in range(6, 8):
                for y in range(5, 21):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')
            for x in range(-4, 6):
                for y in range(5, 7):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')

            for x in range(-21, -2):
                for y in range(-6, -4):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')
            for x in range(-11, -9):
                for y in range(-16, -6):
                    m._mark_cell(m._get_cell([x, y, z]), 'obstacle')

        return m

    def reset(self):
        self.cell_map = self.create_map()
        self.direction = 0  # 0 radians = [1,0], pi/2 radians = [0,1]
        self.position = self.cell_map.get_current_position()
        self.steps = 0
        return self.cell_map.get_local_map()

    def step(self, delta_pos=None, waypoint=None, compass=None):
        """
        Move to a new target position. The target can be given as (dx, dy), (x, y) or (distance, angle)
        :param delta_pos: relative position of next waypoint - tuple-like: (dx, dy)
        :param waypoint: global position of next waypoint - tuple-like: (x, y)
        :param compass: compass representation of next waypoint - tuple-like: (distance, theta), in radians
        :return:
        """
        assert delta_pos is None or waypoint is None or compass is None and waypoint is not compass and \
            waypoint is not delta_pos, 'Check that either delta_pos, waypoint or compass is given'

        if waypoint is not None:
            waypoint = np.concatenate([np.array(waypoint, dtype=float), np.array([0], dtype=float)])  # add z-dim
            distance = np.linalg.norm(waypoint - self.position)

        if delta_pos is not None:
            delta_pos = np.concatenate([np.array(delta_pos, dtype=float), np.array([0], dtype=float)])  # add z-dim
            waypoint = self.position + delta_pos
            distance = np.linalg.norm(waypoint - self.position)

        if compass is not None:
            distance, theta = tuple(compass)
            self.direction = np.unwrap([0, self.direction + theta])[1]
            waypoint = self.position + distance * np.array([np.cos(self.direction), np.sin(self.direction), 0])

        success, num_detected = self.move_to_waypoint(waypoint=waypoint)

        if success:
            reward = num_detected / distance / self.reward_scaling
            done = False
        else:
            reward = -10
            done = True
        obs = self.cell_map.get_local_map()

        self.steps += 1
        if self.steps == self.max_steps:
            done = True
        return obs, reward, done, {}

    def render(self, local=True):
        if local:
            self.cell_map.visualize(num_ticks_approx=20, cell_map=self.cell_map.get_local_map(), ax=self.ax)
        else:
            self.cell_map.visualize(num_ticks_approx=20, ax=self.ax)

    def move_to_waypoint(self, waypoint, step_length=0.1):  # approximately reaches the target

        pos = np.array(self.cell_map.get_current_position(), dtype=np.float32)
        v = np.array(waypoint, dtype=np.float32) - pos
        magnitude = np.linalg.norm(v)
        import math
        if math.isnan(magnitude):
            print("mag is nan. v: ", v)
        v_norm = v / (magnitude + 1e-3)
        num_steps = int(magnitude / step_length)

        success = True
        num_detected_cells = 0
        for step in range(num_steps):
            pos += v_norm * step_length

            if self.cell_map.get_info(pos)['obstacle']:
                success = False
                break
            _, num_detected = self.cell_map.update(pos.copy())
            num_detected_cells += num_detected

        self.position = self.cell_map.get_current_position()

        return success, num_detected_cells

    def close(self):
        pass


if __name__ == '__main__':
    env = MapEnv()
    env.render(local=False)
    #print(env.cell_map.get_info([-22, -22, 0]))
    #print(env.cell_map.get_info([0, 0, 0]))
    #print(env.cell_map.get_info([18, -18, 0]))
    obs, reward, done, _ = env.step(compass=[5, -np.pi/2])
    print("reward 1: " + str(reward) + "    Done: " + str(done))
    env.render()
    time.sleep(1)

    obs, reward, done, _ = env.step(compass=[10, 0])
    print("reward 2: " + str(reward) + "    Done: " + str(done))
    env.render()
    time.sleep(1)

    obs, reward, done, _ = env.step(compass=[10, -3*np.pi/4])
    print("reward 3: " + str(reward) + "    Done: " + str(done))
    env.render()
    time.sleep(1)

    obs, reward, done, _ = env.step(compass=[8, -np.pi/2])
    print("reward 4: " + str(reward) + "    Done: " + str(done))
    env.render()
    time.sleep(1)

    obs, reward, done, _ = env.step(compass=[4, np.pi/2])
    print("reward 5: " + str(reward) + "    Done: " + str(done))
    env.render()

    env.render(local=False)
    time.sleep(10)

    """
    env.move_to_waypoint([18,10,0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)

    env.move_to_waypoint([5, 0, 0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)

    env.move_to_waypoint([-15, 0, 0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)

    env.move_to_waypoint([-5, 15, 0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)

    env.move_to_waypoint([20, 5, 0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)

    env.move_to_waypoint([-10, 0, 0])
    env.cell_map.visualize(num_ticks_approx=20)
    time.sleep(0.04)
    """
