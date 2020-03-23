import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gym
from gym import spaces


def make(**env_kwargs):
    return MapEnv(**env_kwargs)

class MapEnv(gym.Env):

    def __init__(self,
                 starting_map_size=(10., 10., 2.),
                 cell_scale=(1., 1., 1.),
                 starting_position=(0., 0., 0.),
                 buffer_distance=(10., 10., 0.),
                 local_map_dim=(16, 16, 1),
                 vision_range=1,
                 fov_angle=np.pi/2,
                 map_keys=['unknown', 'visible', 'visited', 'obstacle', 'object', 'position'],
                 thresholds={'visible': 1, 'visited': 1, 'obstacle': 1, 'object': 1},
                 map_idx=-1,
                 interactive_plot=False,
                 REWARD_FAILURE=-10,
                 REWARD_STEP=-0.1,
                 ):
        """
        # TODO: improve doc-string
        Creates an n-dimensional map environment, where n is the length of starting_map_size. The map will be centered around
        the starting position
        :param starting_map_size: map size in meters
        :param cell_scale: cell length in meters
        :param starting_position: starting position of the agent
        :param buffer_distance: the furthest distance from border that triggers extension of the map
        :param local_map_dim: number of CELLS in the local map Manually enter starting_map_size and cell_scale to match wanted output
        :param vision_range: vision range in meters. Should probably be smaller than buffer_distance
        :param fov_angle: field of view angle in radians.
        :param map_keys: string of labels for which things to be tracked in the map and visualized
        :param thresholds: dict. Number of detections in a cell needed to mark. as occurance
        :param map_idx: int in range [0,7] to choose from predefined training maps, None to randomize index. -1 default map.
        """
        assert len(starting_map_size) == 3, 'Class does currently only work for exactly 3 dimensions.'
        self.LARGE_NUMBER = 20
        self.REWARD_FAILURE = REWARD_FAILURE
        self.REWARD_STEP = REWARD_STEP
        self.tokens = {'unknown': 1, 'visible': 2, 'visited': 3, 'obstacle': 4, 'object': 5, 'position': 6}
        assert len(starting_map_size) == len(cell_scale) == len(starting_position) == len(buffer_distance) == len(local_map_dim), \
            "Different dimensions discovered in the input"
        assert vision_range >= 0, 'Vision range = {}. Value must be positive'.format(vision_range)
        assert 0 <= fov_angle <= 2*np.pi, 'fov_angle = {}. Value must be in range [0, 2*pi]'.format(fov_angle)
        assert map_keys
        for key in map_keys:
            assert key in self.tokens.keys(), 'Unrecognized map key: {}'.format(key)
        if 'unknown' not in map_keys: map_keys = ['unknown',] + map_keys
        assert len(map_keys) > 1, 'map_keys must contain at least one key that is not \'unknown\''

        self.dimensions = len(starting_map_size)
        for dim in range(self.dimensions):
            assert starting_map_size[dim] > 0, 'map_size[{}] = {}. Value must be > 0'.format(dim, starting_map_size[dim])
            assert cell_scale[dim] > 0, 'cell_scale[{}] = {}. Value must be > 0'.format(dim, cell_scale[dim])
            assert buffer_distance[dim] >= 0, 'buffer_distance[{}] = {}. Value must be >= 0'.format(dim, buffer_distance[dim])
            assert local_map_dim[dim] > 0, 'local_map_dim[{}] = {}. Value must be > 0'.format(dim, local_map_dim[dim])
            if not buffer_distance[dim] >= ((local_map_dim[dim] - 1) * cell_scale[dim] / 2):
                print('WARNING: buffer distance in dimension [{}] might be too short, which can result in a local map of smaller size than intended.'.format(dim))

        # assigned in function reset()
        self.position = None
        self.cell_map = None
        self.map_shape = None
        self.cell_positions = None

        self.starting_map_size = starting_map_size
        self.cell_scale = cell_scale
        self.starting_position = starting_position
        self.buffer_distance = buffer_distance
        self.local_map_dim = local_map_dim
        self.vision_range = vision_range
        self.fov_angle = fov_angle
        self.map_keys = map_keys
        self.thresholds = thresholds
        for key in self.thresholds.keys():  # remove non-valid key-value pairs in thresholds
            if key not in self.map_keys:
                del self.thresholds[key]

        for key in map_keys:  # make sure all thresholds are assigned
            if key in thresholds:
                assert thresholds[key] >= 0, 'Threshold[{}] = {}. Value must be >= 0'.format(key, thresholds[key])
            elif key is not 'unknown' or key is not 'position':
                self.thresholds[key] = 1

        self.reward_scaling = (self.vision_range / self.cell_scale[0]) * \
                              (self.vision_range / self.cell_scale[1]) * np.pi  # divide by area
        # assumes binary observations
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.local_map_dim[0], self.local_map_dim[1],
                                                                  (len(self.map_keys) - 1) * self.local_map_dim[2]),
                                            dtype=np.int)
        self.action_space = spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]),
                                       dtype=np.float64)
        if interactive_plot:  # TODO: double check
            plt.ion()
            fig = plt.figure()
            self.ax = fig.subplots()
        else:
            self.ax = None

    def _create_map(self, map_borders):

        num_cells = np.zeros(self.dimensions, dtype=np.int32)
        buffer_length = np.zeros(self.dimensions)
        # find number of cells that need to be used for each dimension
        for dim in range(self.dimensions):
            length = map_borders[dim][1] - map_borders[dim][0]
            num_cells[dim] = int(np.ceil(length / self.cell_scale[dim]))
            buffer_length[dim] = (num_cells[dim] * self.cell_scale[dim] - length) / 2

        # find (1d-) coordinates describing the edges of the cells
        cell_positions = []
        for i in range(self.dimensions):
            start = map_borders[i][0] - buffer_length[i]
            stop = map_borders[i][1] + buffer_length[i]
            cell_positions.append(np.linspace(start, stop, num_cells[i] + 1))

        cell_map = {}
        for key in self.thresholds.keys():
            cell_map[key] = np.zeros(num_cells, dtype=np.int32)

        map_shape = num_cells

        return cell_map, map_shape, cell_positions

    def _expand(self, size_increase):
        """

        :param size_increase: increment in meters - list of tuple-like e.g. [(5,0), (10,10), (0,1)]
        :return:
        """
        current_borders = [(self.cell_positions[dim][0], self.cell_positions[dim][-1])
                           for dim in range(self.dimensions)]
        num_new_cells = [(int(np.ceil(size_increase[dim][0] / self.cell_scale[dim])),
                          int(np.ceil(size_increase[dim][1] / self.cell_scale[dim]))) for dim in range(self.dimensions)]
        new_borders = [(current_borders[dim][0] - num_new_cells[dim][0] * self.cell_scale[dim],
                        current_borders[dim][1] + num_new_cells[dim][1] * self.cell_scale[dim]) for dim in range(self.dimensions)]
        # create new map
        new_cell_map, new_map_shape, new_cell_positions = self._create_map(new_borders)

        current_shape = self.map_shape
        idx_of_old_map = tuple([slice(num_new_cells[dim][0], current_shape[dim] + num_new_cells[dim][0])
                                for dim in range(self.dimensions)])
        # fill new map with copy of old map
        for key in self.thresholds.keys():
            new_cell_map[key][idx_of_old_map] = self.cell_map[key]

        self.cell_map = new_cell_map
        self.map_shape = new_map_shape
        self.cell_positions = new_cell_positions

    def _automatic_expansion(self, position):
        """
        expand if position (in meters) is within buffer distance of borders.
        :param position:
        :return:
        """
        expand = False
        size_increase = [[0, 0], [0, 0], [0, 0]]
        for dim in range(self.dimensions):
            for j in range(2):
                distance_to_border = np.abs(position[dim] - self.cell_positions[dim][j*-1])
                if distance_to_border < self.buffer_distance[dim]:
                    size_increase[dim][j] = self.buffer_distance[dim]
                    expand = True

        if expand:
            print('expanding map automatically with size: ' + str(size_increase))
            self._expand(size_increase)

    def _get_cell(self, position):

        # check if map needs to be expanded
        expand = False
        size_increase = [[0, 0], [0, 0], [0, 0]]
        for dim in range(self.dimensions):
            if self.cell_positions[dim][0] > position[dim]:
                size_increase[dim][0] = self.cell_positions[dim][0] - position[dim] + self.buffer_distance[dim]
                expand = True
            if self.cell_positions[dim][1] < position[dim]:
                size_increase[dim][0] = position[dim] - self.cell_positions[dim][0] + self.buffer_distance[dim]
                expand = True
        if expand:
            self._expand(size_increase)

        cell = ()
        for dim in range(self.dimensions):
            if position[dim] == self.cell_positions[dim][-1]:  # if at (top) border
                idx = int(np.nonzero(position[dim] > self.cell_positions[dim])[0][-1])
            else:
                idx = int(np.nonzero(position[dim] >= self.cell_positions[dim])[0][-1])
            cell += (idx,)

        return cell

    def _get_position(self, cell):
        """
        Assumes cell is valid.
        :param cell:
        :return:
        """
        positions = [self.cell_positions[dim][cell[dim]] + self.cell_scale[dim] / 2 for dim in range(self.dimensions)]
        return positions

    def _move_to_cell(self, cell):
        if self.cell_map['visited'][cell] < self.LARGE_NUMBER:
            self.cell_map['visited'][cell] += 1

    def _mark_cell(self, cell, map_key):
        if self.cell_map[map_key][cell] < self.LARGE_NUMBER:
            self.cell_map[map_key][cell] += 1

    def _field_of_view(self, position, orientation):
        """

        :param position: absolute position
        :param orientation: normed 2d np array
        :return:
        """
        # currently only works in 2D
        # TODO: generalize to 3D later on
        # approximate function: distance is calculated from cell perspective, not position perspective
        assert 'visible' in self.map_keys, 'no map exists that tracks visible cells'

        center_x = position[0]
        center_y = position[1]
        size_z = self.map_shape[2] * self.cell_scale[2]

        scale_x, scale_y, _ = self.cell_scale
        radius = self.vision_range

        # retrieve x and y coordinates for cell centers
        x = np.expand_dims(self.cell_positions[0][:-1] + self.cell_scale[0] / 2, axis=1)
        y = np.expand_dims(self.cell_positions[1][:-1] + self.cell_scale[1] / 2, axis=0)

        circle_mask = (x-center_x) ** 2 + (y-center_y) ** 2 <= radius ** 2

        cos_theta, sin_theta = np.cos(self.fov_angle / 2), np.sin(self.fov_angle / 2)
        rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        left_border_direction = rot_matrix @ orientation
        right_border_direction = rot_matrix.transpose() @ orientation

        left_mask = left_border_direction[0] * (y-center_y) - left_border_direction[1] * (x-center_x) <= 0
        right_mask = right_border_direction[0] * (y-center_y) - right_border_direction[1] * (x-center_x) >= 0

        mask = circle_mask * ((left_mask & right_mask) if self.fov_angle < np.pi else (left_mask | right_mask))
        # apply same mask on every height (z-dim)
        mask_z = np.repeat(mask[:, :, np.newaxis], size_z, axis=2)

        # mask vision behind obstacles
        visible_obstacles = self.cell_map['obstacle'] * mask_z
        cell_diag = np.linalg.norm([self.cell_scale[0], self.cell_scale[1]])  # length of cell diagonal
        for position_idx in np.argwhere(visible_obstacles):
            xx = self.cell_positions[0][position_idx[0]] + self.cell_scale[0] / 2
            yy = self.cell_positions[1][position_idx[1]] + self.cell_scale[1] / 2

            v = np.array([xx - center_x, yy - center_y])
            # approximation of occlusion lines
            angle = cell_diag / (2 * np.pi * np.linalg.norm(v)) * 5  # TODO: find good factor
            cos_theta, sin_theta = np.cos(angle / 2), np.sin(angle / 2)
            rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            left_border_direction = rot_matrix @ v
            right_border_direction = rot_matrix.transpose() @ v

            left_mask = left_border_direction[0] * (y - center_y) - left_border_direction[1] * (x - center_x) <= 0
            right_mask = right_border_direction[0] * (y - center_y) - right_border_direction[1] * (x - center_x) >= 0
            distance_mask = -v[1] * (y - yy) - v[0] * (x - xx) <= 0
            temp_mask = left_mask * right_mask * distance_mask

            mask = mask * ~temp_mask

        fov_mask = np.repeat(mask[:, :, np.newaxis], size_z, axis=2)

        return fov_mask

    def _update(self, new_position, orientation=None, detected_obstacles=(), detected_objects=()):
        """

        :param new_position: tuple or list, (x, y, z)
        :param detected_obstacles: list of absolute positions
        :param detected_objects: list of absolute positions
        :param orientation: 2d np array of the camera direction (used for _mark_visible()). If None, direction is estimated
        :return: number of cells in field of view that have not yet been detected.
        """
        if orientation is None:  # estimate orientation
            direction = np.array(new_position) - np.array(self.position)
            if np.all(direction == 0):  # avoid zero vector
                direction = np.array([1., 0., 0.])
            orientation = direction[:2] / np.linalg.norm(direction[:2])
        self._automatic_expansion(new_position)  # expand map if we move close to a border
        self._move_to_cell(self._get_cell(new_position))
        self.position = new_position

        for obstacle_position in detected_obstacles:
            self._mark_cell(self._get_cell(obstacle_position), 'obstacle')
        for object_position in detected_objects:
            self._mark_cell(self._get_cell(object_position), 'object')

        fov_mask = self._field_of_view(new_position, orientation)
        # number of detected cells for which the detection threshold has not yet been reached
        num_detected = np.sum(fov_mask) - np.sum(self.cell_map['visible'][fov_mask] // self.thresholds['visible'] == 0)
        self.cell_map['visible'][fov_mask] += 1
        self.cell_map['visible'][fov_mask].clip(0, self.LARGE_NUMBER)

        return num_detected

    def _get_map(self, local=True, binary=True):
        """
        If local, get local map, else get global map.
        If binary = True, binary maps are stacked in z dim, one for each map_key, which is suitable for neural network
        input. Else, the map is built using tokens representing the different keys. Suitable for visualization
        :param binary:
        :return:
        """
        current_cell = self._get_cell(self.position)

        cell_map = self.cell_map.copy()
        # Apply thresholds
        cell_map = {k: np.clip(v // self.thresholds[k], 0, 1) for k, v in cell_map.items()}

        if 'position' in self.map_keys:
            position_map = np.zeros(self.map_shape, dtype=np.int32)
            position_map[current_cell] = 1
            cell_map['position'] = position_map

        if local:
            local_idx = []
            for dim in range(self.dimensions):
                start = np.max((current_cell[dim] - self.local_map_dim[dim] // 2, int(0)))
                end = start + self.local_map_dim[dim]
                local_idx.append(slice(start, end))
            local_idx = tuple(local_idx)
            cell_map = {k: v[local_idx] for k, v in cell_map.items()}

        if binary:
            cell_map = np.concatenate([value for value in cell_map.values()], axis=2)
            cell_map = np.array(cell_map, dtype=np.float32)  # float32 for neural net input
        else:
            temp_cell_map = np.ones(self.local_map_dim, dtype=np.int32) * self.tokens['unknown']
            for key in cell_map.keys():
                temp_idx = np.nonzero(cell_map[key])
                temp_cell_map[temp_idx] = self.tokens[key]
            cell_map = temp_cell_map

        return cell_map

    def _get_info(self, position):
        """
        Get detections for position
        :param position:
        :return:
        """
        return {k: v[self._get_cell(position)] for k, v in self.cell_map.items()}

    def _get_current_position(self):
        return self.position

    def _visualize2d(self, local=True, ax=None, num_ticks_approx=6):
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots()
        assert 2 <= self.dimensions <= 3, 'Map is {}d. Can only visualize 2d and 3d maps.'.format(self.dimensions)

        token_map = self._get_map(local=local, binary=False)
        assert len(token_map.shape) == 3, 'Token map is {}d. (Un-)squeeze to 3d'.format(len(token_map.shape))
        token_map = np.max(token_map, axis=2)

        token_map = token_map.transpose()

        color_dict = {'unknown': 'midnightblue', 'visible': 'lightsteelblue', 'visited': 'limegreen',
                      'obstacle': 'red', 'object': 'gold', 'position': 'darkgreen'}
        color_map = colors.ListedColormap([color_dict[k] for k in self.map_keys])
        bounds = [0,] + [self.tokens[key] for key in self.map_keys]
        norm = colors.BoundaryNorm(bounds, color_map.N)

        ax.imshow(token_map, origin='lower', cmap=color_map, norm=norm)

        if local:
            local_cell_positions = []
            for dim in range(self.dimensions):
                start = self._get_cell(self.position)[dim] - self.local_map_dim[dim] // 2
                start = np.max([start, 0])
                end = start + self.local_map_dim[dim]
                local_cell_positions.append(self.cell_positions[dim][(slice(start, end))])

            x_tick_skip = len(local_cell_positions[0]) // num_ticks_approx + 1
            y_ticks_skip = len(local_cell_positions[1]) // num_ticks_approx + 1
            x_tick_pos = (local_cell_positions[0] - local_cell_positions[0][0] - self.cell_scale[0] / 2) / self.cell_scale[0]
            x_tick_val = local_cell_positions[0]
            y_tick_pos = (local_cell_positions[1] - local_cell_positions[1][0] - self.cell_scale[1] / 2) / self.cell_scale[1]
            y_tick_val = local_cell_positions[1]
        else:
            x_tick_skip = len(self.cell_positions[0]) // num_ticks_approx + 1
            y_ticks_skip = len(self.cell_positions[1]) // num_ticks_approx + 1
            x_tick_pos = (self.cell_positions[0] - self.cell_positions[0][0] - self.cell_scale[0] / 2) / self.cell_scale[0]
            x_tick_val = self.cell_positions[0]
            y_tick_pos = (self.cell_positions[1] - self.cell_positions[1][0] - self.cell_scale[1] / 2) / self.cell_scale[1]
            y_tick_val = self.cell_positions[1]

        ax.set_xticks(x_tick_pos[::x_tick_skip])
        ax.set_xticklabels(x_tick_val[::x_tick_skip])
        ax.set_yticks(y_tick_pos[::y_ticks_skip])
        ax.set_yticklabels(y_tick_val[::y_ticks_skip])

        plt.pause(0.005)

    def _visualize3d(self):  # TODO: implement
        pass

    def _move_to_waypoint(self, waypoint, step_length=0.1):  # naive, straight path
        """
        Reaches target by taking smalls steps until close to target
        TODO: implement 3d
        :param waypoint:
        :param step_length:
        :return:
        """
        success = True
        num_detected_cells = 0

        pos = np.array(self._get_current_position(), dtype=np.float32)
        v = waypoint - pos
        if np.all(v[:2] == 0):  # already at target
            return success, num_detected_cells

        magnitude = np.linalg.norm(v)
        v_norm = v / magnitude
        num_steps = int(magnitude / step_length)

        check_for_obstacles = 'obstacle' in self.map_keys
        for step in range(num_steps):
            pos += v_norm * step_length

            if check_for_obstacles:
                if self._get_info(pos)['obstacle'] > self.thresholds['obstacle']:
                    success = False
                    break
            num_detected_cells += self._update(pos.copy())

        return success, num_detected_cells

    def reset(self, starting_position=None, local=True, binary=True):
        if starting_position is None:
            starting_position = self.starting_position
        self.position = starting_position

        self.starting_map_size
        # describe the edges of the map
        map_borders = [(-length / 2 + offset, length / 2 + offset) for length, offset in
                       zip(self.starting_map_size, starting_position)]
        # create a new map
        self.cell_map, self.map_shape, self.cell_positions = self._create_map(map_borders)
        self._move_to_cell(self._get_cell(self.position))

        return self._get_map(local=local, binary=binary)

    def step(self, action, local=True, binary=True):
        """

        :param action: 2d array describing a waypoint in relative position: (dx, dy)
        :param local:
        :param binary:
        :return:
        """
        delta_position = np.array(action, dtype=np.float32), np.array([0.], dtype=np.float32)
        waypoint = self.position + delta_position

        success, num_detected_cells = self._move_to_waypoint(waypoint)

        if success:
            reward = num_detected_cells / self.reward_scaling - self.REWARD_STEP  # penalizing small steps
            done = False
        else:
            reward = self.REWARD_FAILURE
            done = True
        observation = self._get_map(local=local, binary=binary)
        info = {'env': 'Exploration'}
        return observation, reward, done, info

    def render(self, local=False, num_ticks_approx=6):
        ax = None
        if self.ax is not None:
            ax = self.ax
        self._visualize2d(self, local=local, ax=ax, num_ticks_approx=num_ticks_approx)

    def close(self):
        pass


# TODO New class inheriting from MapEnv, which uses neural actor -
class NeuralMapEnv(MapEnv):
    def __init__(self):
        pass

    def _move_to_waypoint(self, waypoint):
        pass