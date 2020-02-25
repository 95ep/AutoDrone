import numpy as np
import sys
import time
sys.path.append('C:/Users/Filip/Documents/Skola/Exjobb/AutoDrone/Map')
from map_dev import GlobalMap

def make():
    pass

class MapEnv:

    def __init__(self):
        self.cell_map = self.create_map()

    def create_map(self, map_idx=0):

        if map_idx == 0:
            x_size = 51
            y_size = 51
            border_thickness = 5
            z = 0
            m = GlobalMap(map_size=(x_size, y_size, 1),
                          cell_scale=(1, 1, 1),
                          starting_position=(18, -18, z),
                          buffer_distance=(0, 0, 0),
                          local_map_size=(10, 10, 1),
                          detection_threshold_obstacle=1,
                          detection_threshold_object=1,
                          vision_range=3,
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

    def step(self, action):
        """

        :param action: position next waypoint - tuple-like: (x, y)
        :return:
        """

    def move_to_waypoint(self, waypoint, step_length=0.1):  # approximately reaches the target

        pos = np.array(self.cell_map.get_current_position(), dtype=np.float64)
        v = np.array(waypoint) - pos
        magnitude = np.linalg.norm(v)
        v_norm = v / magnitude
        num_steps = int(magnitude / step_length)

        success = True
        for step in range(num_steps):
            pos += v_norm * step_length
            if self.cell_map.get_info(pos)['obstacle']:
                print("CRASHED INTO OBSTACLE")
                success = False
                break
            self.cell_map.update(pos)

        return success


if __name__ == '__main__':
    env = MapEnv()
    env.cell_map.visualize(num_ticks_approx=20)
    #print(env.cell_map.get_info([-22, -22, 0]))
    #print(env.cell_map.get_info([0, 0, 0]))
    #print(env.cell_map.get_info([18, -18, 0]))
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