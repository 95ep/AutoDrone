import gym
from gym import spaces
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch
import vtkplotter

from Environments.env_utils import make_env_utils
from NeuralNetwork.neural_net import NeuralNet


def make(**env_kwargs):
    """
    Can be used instead of calling MapEnv() directly.
    """
    if 'env_str' in env_kwargs:
        if env_kwargs['env_str'] == 'AutonomousDrone':
            return AirSimMapEnv(**env_kwargs)
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
                 map_idx=0,
                 interactive_plot=False,
                 REWARD_FAILURE=-10,
                 REWARD_STEP=-0.5,
                 ):
        """
        Creates an n-dimensional voxel/cell map environment, where n is decided by len(starting_map_size).
        The map will be centered around the starting position. The map will dynamically increase in size when necessary.
        The map can keep track of locations for different categories specified by map_keys. A local map can be accessed
        which only shows a part of the full map, centered around the current position of the agent. The map can also be
        used as a reinforcement learning environment, which follows the gym framework.

        :param starting_map_size: map size in meters
        :param cell_scale: length of each cell in meters
        :param starting_position: starting position of the agent
        :param buffer_distance: the furthest distance from a border that triggers extension of the map
        :param local_map_dim: number of CELLS in the local map. Manually enter starting_map_size and cell_scale to match wanted output shape.
        :param vision_range: vision range in meters. Recommended that it is smaller than buffer_distance, but not necessary
        :param fov_angle: field of view angle of the vision of the drone, in radians.
        :param map_keys: string of labels for which things to be tracked in the map and visualized
        :param thresholds: dict. Number of detections in a cell needed to mark as occurrence
        :param map_idx: int in range [-1,9] to choose from predefined training maps, -1 to randomize index. 0 default empty map, suitable for Airsim.
        :param interactive_plot: Set to True only if rendering 2D pyplots and want the plot to update in the same window.
        :param REWARD_FAILURE: Reinforcement learning penalty for collisions
        :param REWARD_STEP: Reinforcement learning penalty for defining a waypoint
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
        keys = list(self.thresholds.keys())
        for key in keys:  # remove non-valid key-value pairs in thresholds
            if key not in self.map_keys:
                del self.thresholds[key]

        for key in map_keys:  # make sure all thresholds are assigned
            if key in thresholds:
                assert thresholds[key] >= 0, 'Threshold[{}] = {}. Value must be >= 0'.format(key, thresholds[key])
            elif key is not 'unknown' or key is not 'position':
                self.thresholds[key] = 1
        # detecting 100% of vision area approximately equals reward of 1
        self.reward_scaling = (self.vision_range / self.cell_scale[0]) * \
                              (self.vision_range / self.cell_scale[1]) * fov_angle / 2  # divide by area
        # assumes binary observations
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.local_map_dim[0], self.local_map_dim[1],
                                                                  (len(self.map_keys) - 1) * self.local_map_dim[2]),
                                            dtype=np.int)
        self.action_space = spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]),
                                       dtype=np.float64)
        if interactive_plot:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.subplots()
        else:
            self.fig = None
            self.ax = None
        self.map_idx = map_idx
        self.z_value = None
        self.total_detected = None
        self.solved_threshold = None

    def _debug(self, msg=""):
        if msg:
            print("===== " + msg + " debug start =====")
        print("Position: ", self.position)
        print("Map shape: ", self.map_shape)
        print("Borders:   x:["+str(self.cell_positions[0][0])+", " + str(self.cell_positions[0][-1]) +"], " +
                         "y:["+str(self.cell_positions[1][0])+", " + str(self.cell_positions[1][-1]) +"], " +
                         "z:["+str(self.cell_positions[2][0])+", " + str(self.cell_positions[2][-1]) +"]")
        print("Local map dimension: ", self.local_map_dim)
        print("cell map shape: ", self.cell_map['visited'].shape, self.cell_map['obstacle'].shape)
        print("observation shape: ", self._get_map(local=True, binary=True).shape)
        if msg:
            print("===== " + msg + " debug end =====")
            print("")


    def _create_map(self, map_borders):
        """
        Creates an empty map.

        :param map_borders: list with start and end positions in each dimension. Ex: [[x0, x1],[y0,y1]]
        :return: dictionary of cell/voxel maps for each map key, size of map, array of cell positions for each dimension.
        """
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
        Increase the size of the map.
        :param size_increase: increment in meters - list of tuple-like e.g. [(5,0), (10,10), (0,1)]
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
        Find the size to expand the map with. Expand only if position (in meters)
        is outside borders or within the buffer distance of borders.
        :param position: array-like absolute position, or list of array-like
        """
        position = np.array(position, dtype=np.float32)
        if len(position.shape) == 1:
            position = np.expand_dims(position, axis=0)
        expand = False
        size_increase = [[0, 0], [0, 0], [0, 0]]
        for pos in position:
            for dim in range(self.dimensions):
                if self.cell_positions[dim][0] + self.buffer_distance[dim] > pos[dim]:
                    temp_increase = self.cell_positions[dim][0] - pos[dim] + self.buffer_distance[dim]
                    temp_increase = np.max((temp_increase, self.buffer_distance[dim]))
                    size_increase[dim][0] = np.max((temp_increase, size_increase[dim][0]))
                    expand = True
                if self.cell_positions[dim][-1] - self.buffer_distance[dim] < pos[dim]:
                    temp_increase = pos[dim] - self.cell_positions[dim][-1] + self.buffer_distance[dim]
                    temp_increase = np.max((temp_increase, self.buffer_distance[dim]))
                    size_increase[dim][1] = np.max((temp_increase, size_increase[dim][1]))
                    expand = True

        if expand:
            self._expand(size_increase)

    def _get_cell(self, position):
        """
        Get index to the cell at position
        :param position: array like [x, y, ...]
        :return: idx of cell (ix,iy,...)
        """
        self._automatic_expansion(position)
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
        Get the position of the cell. Assumes cell is valid (exists in map).
        :param cell: idx of cell (ix,iy,...)
        :return: position of cell, array like [x, y, ...]
        """
        positions = [self.cell_positions[dim][cell[dim]] + self.cell_scale[dim] / 2 for dim in range(self.dimensions)]
        return positions

    def _move_to_cell(self, cell):
        """
        Moves agent position to the position of the cell
        :param cell: idx of cell (ix,iy,...)
        """
        if 'visited' in self.map_keys:
            if self.cell_map['visited'][cell] < self.LARGE_NUMBER:
                self.cell_map['visited'][cell] += 1

    def _mark_cell(self, cell, map_key):
        """
        Adds a detection of the map_key category in specific cell.
        :param cell: idx of cell (ix,iy,...)
        :param map_key: the detected category, e.g. 'object'
        """
        if self.cell_map[map_key][cell] < self.LARGE_NUMBER:
            self.cell_map[map_key][cell] += 1

    def _field_of_view(self, position, orientation, z_value=None):
        """
        Vision model of the agent. Find which cells in the map that are visible to the agent given the position and
        orientation. Obstacles that are present in the map will occlude cells behind them.

        :param position: absolute position
        :param orientation: 2d np array (v_x, v_y). (It will be normalized)
        :param z_value: None or float. If float, generates vision cone with height=1 at z_value suitable for 2d exploration training.
        :return:
        """
        # approximate function: distance is calculated from cell perspective, not position perspective
        assert 'visible' in self.map_keys, 'no map exists that tracks visible cells'
        assert len(orientation) == 2, 'orientation should be 2 dimensional vector (v_x, v_y)'
        assert orientation[0] ** 2 + orientation[1] ** 2 > 0, 'orientation must not be zero vector'
        orientation = np.array(orientation, dtype=np.float32)
        orientation /= np.linalg.norm(orientation)
        center_x = position[0]
        center_y = position[1]
        center_z = position[2]

        scale_x, scale_y, scale_z = self.cell_scale
        radius = self.vision_range

        # retrieve x and y coordinates for cell centers
        x = np.reshape(self.cell_positions[0][:-1] + scale_x / 2, (-1, 1, 1))
        y = np.reshape(self.cell_positions[1][:-1] + scale_y / 2, (1, -1, 1))
        z = np.reshape(self.cell_positions[2][:-1] + scale_z / 2, (1, 1, -1))

        mask_sphere = (x-center_x) ** 2 + (y-center_y) ** 2 + (z-center_z) ** 2 <= radius ** 2

        cos_theta, sin_theta = np.cos(self.fov_angle / 2), np.sin(self.fov_angle / 2)
        rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # find first mask of vision cone in the x-y plane
        v_left = rot_matrix @ orientation
        v_right = rot_matrix.transpose() @ orientation
        mask_left = v_left[0] * (y-center_y) - v_left[1] * (x-center_x) + z - z <= 0  # '+z-z' to make mask 3d
        mask_right = v_right[0] * (y-center_y) - v_right[1] * (x-center_x) + z - z >= 0

        if z_value is not None:
            cone_mask = mask_left & mask_right if self.fov_angle < np.pi else mask_left | mask_right
            z_mask_upper = x - x + y - y + z <= z_value + 0.05  # adding small tolerance
            z_mask_lower = x - x + y - y + z >= z_value - 0.05  # adding small tolerance
            mask = mask_sphere * cone_mask * z_mask_upper * z_mask_lower
        else:
            # find second mask of vision cone in the orientation-z plane
            temp_up = rot_matrix @ np.array([1, 0])  # rotated vector in the orientation-z plane
            v_up = np.array([orientation[0] * temp_up[0], orientation[1] * temp_up[0], temp_up[1]])
            # approximately cross(v, rotation(pi/2) * v), find perpendicular normal vector
            n_up = np.array([-v_up[0]*v_up[2], -v_up[1]*v_up[2], v_up[0] ** 2 + v_up[1] ** 2])
            # mask points on the wrong side of the plane desribed by the normal vector
            mask_up = n_up[0] * (x-center_x) + n_up[1] * (y-center_y) + n_up[2] * (z-center_z) <= 0
            # repeat for other border of the cone
            temp_down = rot_matrix.transpose() @ np.array([1, 0])
            v_down = np.array([orientation[0] * temp_down[0], orientation[1] * temp_down[0], temp_down[1]])
            n_down = np.array([-v_down[0]*v_down[2], -v_down[1]*v_down[2], v_down[0] ** 2 + v_down[1] ** 2])
            mask_down = n_down[0] * (x-center_x) + n_down[1] * (y-center_y) + n_down[2] * (z-center_z) >= 0

            cone_mask = (mask_left & mask_right) * (mask_up & mask_down) if self.fov_angle < np.pi else (mask_left | mask_right) * (mask_up | mask_down)
            mask = mask_sphere * cone_mask

        # mask vision behind obstacles
        visible_obstacles = self.cell_map['obstacle'] * mask
        cell_diag = np.linalg.norm(np.array(self.cell_scale))  # length of cell diagonal ~= maximum occlusion size

        for position_idx in np.argwhere(visible_obstacles):
            xx = self.cell_positions[0][position_idx[0]] + scale_x / 2
            yy = self.cell_positions[1][position_idx[1]] + scale_y / 2
            zz = self.cell_positions[2][position_idx[2]] + scale_z / 2

            v = np.array([xx - center_x, yy - center_y, zz - center_z])
            # approximation of occlusion lines
            angle = np.arctan(cell_diag / np.linalg.norm(v)) * 1
            angle = np.min((angle, np.pi))
            cos_theta, sin_theta = np.cos(angle / 2), np.sin(angle / 2)
            rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            v_left = rot_matrix @ v[:2]
            v_right = rot_matrix.transpose() @ v[:2]

            mask_left = v_left[0] * (y - center_y) - v_left[1] * (x - center_x) + z - z <= 0
            mask_right = v_right[0] * (y - center_y) - v_right[1] * (x - center_x) + z - z >= 0

            norm_v_2d = np.linalg.norm(v[:2])
            temp_up = rot_matrix @ np.array([norm_v_2d, v[2]])
            v_up = np.array([v[0] * temp_up[0] / norm_v_2d, v[1] * temp_up[0] / norm_v_2d, temp_up[1]])
            n_up = np.array([-v_up[0]*v_up[2], -v_up[1]*v_up[2], v_up[0] ** 2 + v_up[1] ** 2])
            mask_up = n_up[0] * (x-center_x) + n_up[1] * (y-center_y) + n_up[2] * (z-center_z) <= 0
            temp_down = rot_matrix.transpose() @ np.array([norm_v_2d, v[2]])
            v_down = np.array([v[0] * temp_down[0] / norm_v_2d, v[1] * temp_down[0] / norm_v_2d, temp_down[1]])
            n_down = np.array([-v_down[0]*v_down[2], -v_down[1]*v_down[2], v_down[0] ** 2 + v_down[1] ** 2])
            mask_down = n_down[0] * (x-center_x) + n_down[1] * (y-center_y) + n_down[2] * (z-center_z) >= 0

            mask_distance = - v[0] * (x - xx) - v[1] * (y - yy) - v[2] * (z - zz) <= 0
            temp_mask = mask_left * mask_right * mask_up * mask_down * mask_distance
            mask *= ~temp_mask

        fov_mask = mask
        return fov_mask

    def _update(self, new_position, orientation=None, detected_obstacles=(), detected_objects=(), z_value=None):
        """
        Update the environment according to the new position and orientation, and add detections to the map.

        :param new_position: tuple or list, (x, y, z)
        :param orientation: 2d np array of the camera direction (used for _mark_visible()). If None, direction is estimated
        :param detected_obstacles: list of absolute positions
        :param detected_objects: list of absolute positions
        :param z_value: specify the height for 2D exploration, else None.
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

        fov_mask = self._field_of_view(new_position, orientation, z_value=z_value)
        # number of detected cells for which the detection threshold has not yet been reached
        num_detected = np.sum(fov_mask) - np.sum((self.cell_map['visible'][fov_mask] // self.thresholds['visible']) > 0)
        self.cell_map['visible'][fov_mask] += 1
        self.cell_map['visible'][fov_mask].clip(0, self.LARGE_NUMBER)

        return num_detected

    def _get_map(self, local=True, binary=True):
        """
        If local, get local map, else get global map.
        If binary = True, binary maps are stacked in z dim, one for each map_key, which is suitable for neural network
        input. Else, the map is built using tokens representing the different keys. Suitable for visualization
        """
        current_cell = self._get_cell(self.position)
        # Expand map if local map covers non-existing cells
        if local:
            positions = []
            for dim in range(self.dimensions):
                temp_pos = self.position
                temp_distance = np.zeros((self.dimensions), dtype=np.float32)
                temp_distance[dim] = np.ceil(self.local_map_dim[dim] / 2) * self.cell_scale[dim]
                positions.append(temp_pos + temp_distance)
                positions.append(temp_pos - temp_distance)
            self._automatic_expansion(positions)

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
                start = current_cell[dim] - self.local_map_dim[dim] // 2
                end = start + self.local_map_dim[dim]
                local_idx.append(slice(start, end))
            local_idx = tuple(local_idx)
            cell_map = {k: v[local_idx] for k, v in cell_map.items()}
            map_shape = self.local_map_dim
        else:
            map_shape = self.map_shape

        if binary:
            cell_map = np.concatenate([cell_map[key] for key in cell_map.keys() if (key != 'unknown')], axis=2)
            cell_map = np.array(cell_map, dtype=np.float32)  # float32 for neural net input
        else:
            temp_cell_map = np.ones(map_shape, dtype=np.int32) * self.tokens['unknown']
            for key in cell_map.keys():
                temp_idx = np.nonzero(cell_map[key])
                temp_cell_map[temp_idx] = self.tokens[key]
            cell_map = temp_cell_map

        return cell_map

    def _get_info(self, position):
        """
        Get detections for position
        """
        return {k: v[self._get_cell(position)] for k, v in self.cell_map.items()}

    def _get_current_position(self):
        return self.position

    def _crop_token_map(self, local, ceiling_z=None, floor_z=None):
        """
        Crop the map, removing cells outside the ceiling/floor values.

        :param local: Also crop in x,y dimensions.
        """
        token_map = self._get_map(local=local, binary=False)
        if ceiling_z is not None:
            assert ceiling_z > self.cell_positions[2][0], 'Ceiling threshold set too low. Must be bigger than {}'.format(self.cell_positions[2][0])
            if ceiling_z > self.cell_positions[2][-1]:
                idx = -1
            else:
                idx = -1 + np.argwhere(self.cell_positions[2] >= ceiling_z)[0][0]
            token_map = token_map[:, :, :idx]

        if floor_z is not None:
            assert floor_z < self.cell_positions[2][-1], 'Floor threshold set too high. Must be smaller than {}'.format(self.cell_positions[2][-1])
            if floor_z < self.cell_positions[2][0]:
                idx = 0
            else:
                idx = 1 + np.argwhere(self.cell_positions[2] <= floor_z)[-1][0]
            token_map = token_map[:, :, idx:]
        return token_map

    def _visualize_2d(self, token_map, local=False, ax=None, num_ticks_approx=6):
        """
        Plot a 2D visualization of the map

        :param token_map: the map to be visualized
        :param local: if True, show cropped part of the map, else show everything
        :param ax: pyplot object to show the plot in
        :param num_ticks_approx: Number of ticks on the axis in the plot
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots()
        assert 2 <= self.dimensions <= 3, 'Map is {}d. Can only visualize 2d and 3d maps.'.format(self.dimensions)

        assert len(token_map.shape) == 3, 'Token map is {}d. (Un-)squeeze to 3d'.format(len(token_map.shape))
        token_map = np.max(token_map, axis=2)

        token_map = token_map.transpose()

        color_dict = {'unknown': 'midnightblue', 'visible': 'lightsteelblue', 'visited': 'limegreen',
                      'obstacle': 'red', 'object': 'gold', 'position': 'darkgreen'}
        color_map = colors.ListedColormap([color_dict[k] for k in self.map_keys])
        bounds = [0.5] + [self.tokens[key] + 0.5 for key in self.map_keys]
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

    def _visualize_3d(self, token_map, show_detected=False, voxels=False):
        """
        Plot a 3D visualization of the map.

        :param token_map: the map to be visualized.
        :param show_detected: Show voxels that are 'visited' if True.
        :param voxels: Show cubic voxels if True, else show 3D points.
        """
        if not voxels:
            X, Y, Z = np.mgrid[:token_map.shape[0], :token_map.shape[1], :token_map.shape[2]]
            point_list = np.array([[x, y, z] for x, y, z in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])  # 'flat' (2d) index array
            colors = np.array(['black', 'midnightblue', 'lightsteelblue', 'limegreen',
                               'red', 'gold', 'darkgreen'])  # black color to shift index + 1
            alphas = np.array([0, 0.05, 0.15, 0.6, 0.6, 1, 1])
            token_map_flat = np.ravel(token_map)
            color_list = colors[token_map_flat]
            alpha_list = alphas[token_map_flat]
            high_pass_filter_threshold = 1.5 if show_detected else 2.5
            mask = token_map_flat > high_pass_filter_threshold

            r = 25  # point radius
            points = vtkplotter.shapes.Points(point_list[mask], r=r, c=color_list[mask], alpha=alpha_list[mask])
            vtkplotter.show(points, interactive=True, newPlotter=True, axes={'xyGrid':True, 'yzGrid': True, 'zxGrid':True})
        else:
            factor = 2
            new_map = np.zeros((token_map.shape[0] * factor, token_map.shape[1] * factor, token_map.shape[2] * factor))
            for i in range(factor):
                for j in range(factor):
                    for k in range(factor):
                        new_map[i::factor, j::factor, k::factor] = token_map
            token_map = new_map

            color_list = np.array([[0.2, 1., 0.2, 0.8], [1., 0.2, 0.2, 0.6], [1., 1., 0, 1.], [0., 0.7, 0., 1.]])
            if show_detected: color_list = np.concatenate((np.array([[0.9, 0.95, 1., 0.2]]), color_list))
            color_map = ListedColormap(color_list)
            vol = vtkplotter.Volume(token_map)  # , c=colors, alpha=alphas)
            lego = vol.legosurface(vmin=2 if show_detected else 3, vmax=6, cmap=color_map)

            vtkplotter.show(lego, interactive=True, newPlotter=True)

    def _training_map(self, map_idx):
        """
        Obtain toy map for 2D exploration
        """
        from Environments.Exploration import training_maps
        return training_maps.generate_map(map_idx)

    def _move_by_delta_position(self, delta_position, step_length=0.1, safe_mode=False):  # naive, straight path
        """
        For 2D exploration: Move agent. Reaches target by taking smalls steps until close to target
        :param delta_position: (dx, dy, 0) for the time being
        :param step_length: length of each step in meters
        :param safe_mode: If true, the movement terminates before an obstacle is hit
        :return: success - False if collision occured, num_detected_cells - number of cells that were seen, steps taken, done - True if trajectory over (collision or max steps)
        """
        success = True
        done = False
        num_detected_cells = 0
        steps = 0

        pos = np.array(self._get_current_position(), dtype=np.float32)
        v = delta_position
        if np.all(v == 0):  # already at target
            return success, num_detected_cells, steps, done

        magnitude = np.linalg.norm(v)
        v_norm = v / magnitude
        num_steps = int(magnitude / step_length)

        check_for_obstacles = 'obstacle' in self.map_keys
        for step in range(num_steps):
            pos += v_norm * step_length

            if check_for_obstacles:
                if self._get_info(pos)['obstacle'] >= self.thresholds['obstacle']:
                    if safe_mode:
                        success = True
                        done = False
                    else:
                        success = False
                        done = True
                    break
            num_detected_cells += self._update(pos.copy(), z_value=self.z_value)
            steps += 1

        return success, num_detected_cells, steps, done

    def reset(self, starting_position=None, local=True, binary=True):
        """
        Function according to reinforcement learning framework. Creates a new empty map and returns an observation of it.

        :param starting_position: position of agent.
        :param local: boolean, local or global observation (cropped or full)
        :param binary: One cell map for each map keys is produced if True, else one map in total with different values showing the map keys.
        :return: map observation
        """
        self.total_detected = 0
        if self.map_idx != 0:
            starting_position, obstacles, self.solved_threshold = self._training_map(self.map_idx)
            self.z_value = starting_position[2]

        if starting_position is None:
            starting_position = self.starting_position
        self.position = starting_position

        # describe the edges of the map
        map_borders = [(-length / 2 + offset, length / 2 + offset) for length, offset in
                       zip(self.starting_map_size, starting_position)]
        # create a new map
        self.cell_map, self.map_shape, self.cell_positions = self._create_map(map_borders)
        self._automatic_expansion(self.position)  # expand map if we move close to a border
        self._move_to_cell(self._get_cell(self.position))
        if self.map_idx != 0:
            self._update(self.position, detected_obstacles=obstacles)
            self.cell_map['visible'][:] = 0
        observation = self._get_map(local=local, binary=binary)
        return observation

    def step(self, action, local=True, binary=True, safe_mode=False):
        """
        Function according to reinforcement learning framework. For 2D exploration. Moves the agent to the waypoint defined by action.

        :param action: 2d array describing a waypoint in relative position: (dx, dy)
        :return: observation, reward, done, info
        """
        delta_position = np.concatenate((np.array(action, dtype=np.float32), np.array([0.], dtype=np.float32)), axis=0)
        terminated_at_target, num_detected_cells, steps, done = self._move_by_delta_position(delta_position, safe_mode=safe_mode)

        steps = np.max((steps, 1))
        reward = 25 * num_detected_cells / self.reward_scaling / steps
        if steps < 10:
            reward *= (steps / 10) ** 4
        else:
            reward *= (steps / 10) ** (1/4)
        if done:
            reward = self.REWARD_FAILURE

        observation = self._get_map(local=local, binary=binary)
        info = {'env': 'Exploration', 'terminated_at_target': terminated_at_target}

        self.total_detected += num_detected_cells
        info['explored'] = self.total_detected
        if self.solved_threshold is not None:
            if self.total_detected >= self.solved_threshold:
                done = True
                info['solved'] = True

        return observation, reward, done, info

    def render(self, render_3d=False, local=False, num_ticks_approx=6, show_detected=False, voxels=True, ceiling_z=None, floor_z=None):
        """
        Calls the different visualization functions to render the environement state.
        :param render_3d: True, else 2D render
        :param local: If True, crop the map to only include the immediate surroundings.
        :param num_ticks_approx:
        :param show_detected: Show cells that are 'visited' if True.
        :param voxels: Plot using cubic voxels, else 3D points
        :param ceiling_z: crop the map at specified height, float.
        :param floor_z: crop the map at specified height, float.
        :return:
        """
        token_map = self._crop_token_map(local, ceiling_z=ceiling_z, floor_z=floor_z)
        if render_3d:
            self._visualize_3d(token_map, show_detected=show_detected, voxels=voxels)
        else:
            ax = None
            if self.ax is not None:
                ax = self.ax
            self._visualize_2d(token_map, local=local, ax=ax, num_ticks_approx=num_ticks_approx)


    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        del self


class AirSimMapEnv(MapEnv):
    """
    Extension of MapEnv which includes AirSim. Reimplements 'move_by_delta_position' to use a local navigation agent instead of
    the naive approach to move straight towards the goal.
    """
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
                 map_idx=0,
                 interactive_plot=False,
                 REWARD_FAILURE=-10,
                 REWARD_STEP=-0.5,
                 **parameters,
                 ):

        super().__init__(starting_map_size=starting_map_size,
                         cell_scale=cell_scale,
                         starting_position=starting_position,
                         buffer_distance=buffer_distance,
                         local_map_dim=local_map_dim,
                         vision_range=vision_range,
                         fov_angle=fov_angle,
                         map_keys=map_keys,
                         thresholds=thresholds,
                         map_idx=map_idx,
                         interactive_plot=interactive_plot,
                         REWARD_FAILURE=REWARD_FAILURE,
                         REWARD_STEP=REWARD_STEP,
                         )

        parameters_airsim = {'env_str': 'AirSim'}
        parameters_airsim['AirSim'] = parameters['AirSim']
        self.env_utils_airsim, self.env_airsim = make_env_utils(**parameters_airsim)
        network_kwargs = self.env_utils_airsim.get_network_kwargs()
        network_kwargs.update(parameters['Exploration']['local_navigation']['neural_network'])  # add additional kwargs from parameter file

        self.local_navigator = NeuralNet(**network_kwargs)
        self.local_navigator.load_state_dict(torch.load(parameters['Exploration']['local_navigation']['weights']))

        self.navigator_max_steps = parameters['navigator_max_steps']
        self.object_detection_frequency = parameters['object_detection_frequency']
        self.obstacle_detection_frequency = parameters['obstacle_detection_frequency']
        self.env_airsim.setup_object_detection(**parameters['object_detection'])

        self.reward_scaling = (self.vision_range / self.cell_scale[0]) * \
                              (self.vision_range / self.cell_scale[1]) * \
                              (self.vision_range / self.cell_scale[2]) * fov_angle / 2  # divide by area

    def _move_by_delta_position(self, delta_position, safe_mode=None):
        self.env_airsim.target_position = self._get_current_position() + delta_position
        self.env_airsim.valid_trgt = True

        obs_air = self.env_airsim._get_state()
        done, trajectory_ended, collision = False, False, False
        success = False
        num_detected_cells = 0
        steps = 0
        # move to waypoint
        while not trajectory_ended:

            obs_vector, obs_visual = self.env_utils_airsim.process_obs(obs_air)
            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
            with torch.no_grad():
                value, action, log_prob = self.local_navigator.act(comb_obs, deterministic=False)
            action = self.env_utils_airsim.process_action(action)
            obs_air, reward, collision, info = self.env_airsim.step(action)
            if collision:
                done = True
                trajectory_ended = True
            elif action == 0:
                done = False
                if safe_mode:
                    success = True
                else:
                    success = info['terminated_at_target']
                trajectory_ended = True
            elif steps == self.navigator_max_steps:
                done = False
                success = False
                trajectory_ended = True

            object_positions = []
            obstacle_positions = []
            if steps > 0 and steps % self.object_detection_frequency == 0:
                object_positions = self.env_airsim.get_trgt_objects()
                if len(object_positions) > 0:
                    print("Number of objects found {}".format(len(object_positions)))
            if steps > 0 and steps % self.obstacle_detection_frequency == 0:
                obstacle_positions = self.env_airsim.get_obstacles(field_of_view=self.fov_angle)
            pos = self.env_airsim.get_position()
            orientation = self.env_airsim.get_orientation()
            num_detected_cells += self._update(pos.copy(),
                                               orientation=orientation,
                                               detected_objects=object_positions,
                                               detected_obstacles=obstacle_positions)
            steps += 1

        return success, num_detected_cells, steps, done

    def reset(self):
        _ = self.env_airsim.reset()
        starting_position = self.env_airsim.get_position()
        observation = super().reset(starting_position=starting_position, local=True, binary=True)
        return observation
