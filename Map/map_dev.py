import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class GlobalMap:
    # Channel structure:
    # 0 visited
    # 1 obstacle present
    # 2 target object present
    def __init__(self,
                 map_size=(10, 10, 2),
                 cell_scale=(1, 1, 1),
                 starting_position=(0, 0, 0.5),
                 buffer_distance=(10, 10, 0),
                 local_map_size=(5, 5, 1),
                 detection_threshold_obstacle=5,
                 detection_threshold_object=2,
                 vision_range=0,
                 fov_angle=2*np.pi,
                 ):
        """
        OBS! Do not report on obstacles/objects further away than the buffer distance.
             Size of local map should be smaller than 0.5*buffer_distance/cell_scale.
        :param map_size: map size in meters
        :param cell_scale: cell length in meters
        :param starting_position: starting position of the agent
        :param buffer_distance: the furthest distance from border that triggers extension of the map
        :param local_map_size: number of CELLS in the local map. Manually enter map_size and cell_scale to match wanted output
        :param vision_range: vision range in meters. Should probably be smaller than buffer_distance
        :param fov_angle: field of view angle in radians. TODO: implement
        """
        assert len(map_size) == len(cell_scale) == len(starting_position) == len(buffer_distance), "different dimensions discovered in the input"
        self.dimensions = len(map_size)
        for dim in range(self.dimensions):
            if dim is not 2:
                if not 2*buffer_distance[dim] >= local_map_size[dim]:
                    print('WARNING: buffer distance in dimension [{}] might be too short, which can result in a local map of smaller size than intended.'.format(dim))
        self.cell_scale = cell_scale
        self.position = starting_position
        self.buffer_distance = buffer_distance
        self.local_map_size = local_map_size
        self.vision_range = vision_range
        self.thresholds = {'visible': 1, 'visited': 1, 'obstacle': detection_threshold_obstacle, 'object': detection_threshold_object}

        map_borders = [(-length/2, length/2) for length in map_size]
        if self.dimensions == 3:
            map_borders[2] = (0.0, float(map_size[2]))

        valid_start_pos = [map_borders[i][0] <= starting_position[i] <= map_borders[i][1] for i in range(self.dimensions)]
        assert all(valid_start_pos), 'starting position ' + str(starting_position) + ' is outside map with borders ' + str(map_borders) + ']'

        self.map_size = map_size
        self.cell_map, self.cell_map_shape, self.cell_positions = self._create_map(map_borders)
        self.update(starting_position)

    def _create_map(self, map_borders):

        num_cells = np.zeros(self.dimensions, dtype=int)
        buffer_length = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            length = map_borders[i][1] - map_borders[i][0]
            num_cells[i] = int(np.ceil(length / self.cell_scale[i]))
            assert num_cells[i] > 0, 'non-positive length in dimension {}'.format(i)
            buffer_length[i] = (num_cells[i] * self.cell_scale[i] - length) / 2

        #print("num_cells: " + str(num_cells))
        cell_positions = []
        for i in range(self.dimensions):
            start = map_borders[i][0] - buffer_length[i]
            stop = map_borders[i][1] + buffer_length[i]
            cell_positions.append(np.linspace(start, stop, num_cells[i] + 1))

        self.map_size = num_cells
        visible_map = np.zeros(num_cells, dtype=int)
        visited_map = np.zeros(num_cells, dtype=int)
        obstacle_map = np.zeros(num_cells, dtype=int)
        object_map = np.zeros(num_cells, dtype=int)
        cell_map = {'visible': visible_map, 'visited': visited_map, 'obstacle': obstacle_map, 'object': object_map}
        cell_map_shape = num_cells

        first_and_last = [[c[0], c[-1]] for c in cell_positions]
        print('Created new map with borders: ' + str(map_borders) + '. Resulting cell borders:' + str(first_and_last))
        return cell_map, cell_map_shape, cell_positions

    def _expand(self, size_increase):
        #print("self.pos: " + str(self.position))
        #print("self cell: " + str(np.nonzero(self.cell_map)))
        current_borders = [(self.cell_positions[i][0], self.cell_positions[i][-1])
                           for i in range(self.dimensions)]
        num_new_cells = [(int(np.ceil(size_increase[i][0] / self.cell_scale[i])),
                          int(np.ceil(size_increase[i][1] / self.cell_scale[i]))) for i in range(self.dimensions)]
        new_borders = [(current_borders[i][0] - num_new_cells[i][0] * self.cell_scale[i],
                        current_borders[i][1] + num_new_cells[i][1] * self.cell_scale[i]) for i in range(self.dimensions)]

        new_cell_map, new_cell_map_shape, new_cell_positions = self._create_map(new_borders)

        current_shape = self.cell_map_shape
        idx_of_old_map = tuple([slice(num_new_cells[i][0], current_shape[i] + num_new_cells[i][0]) for i in range(self.dimensions)])
        #print("idx: " + str(idx_of_old_map))
        for k, v in new_cell_map.items():
            new_cell_map[k][idx_of_old_map] = self.cell_map[k]

        self.cell_map = new_cell_map
        self.cell_map_shape = new_cell_map_shape
        self.cell_positions = new_cell_positions
        #print("self.pos: " + str(self.position))
        #print("self cell: " + str(np.nonzero(self.cell_map)))

    def _get_cell(self, position):
        cell = ()
        for i in range(3):
            #print('[dim]: {},  cel_pos[0]: {}, pos: {}, cel_pos[-1]: {}'.format(i, self.cell_positions[i][0], position[i], self.cell_positions[i][-1]))
            assert self.cell_positions[i][0] <= position[i] <= self.cell_positions[i][-1], 'trying to get cell index for position {} lying outside the map'.format(position)
            if position[i] == self.cell_positions[i][-1]:  # if at (top) border
                idx = int(np.nonzero(position[i] > self.cell_positions[i])[0][-1])
            else:
                idx = int(np.nonzero(position[i] >= self.cell_positions[i])[0][-1])
            cell += (idx,)

        return cell

    def _get_position(self, cell_idx):
        positions = [self.cell_positions[i][cell_idx[i]] + self.cell_scale[i] / 2 for i in range(self.dimensions)]
        return positions

    def _visit_cell(self, cell):
        self.cell_map['visited'][cell] = 1

    def _mark_cell(self, cell, label):
        if self.cell_map[label][cell] < 100: # Arbitrary large number
            self.cell_map[label][cell] += 1
            #print("MARKING {} in cell {}. Value = {}".format(label, cell, self.cell_map[label][cell]))

    def _automatic_expansion(self, position):
        expand = False
        size_increase = [[0, 0], [0, 0], [0, 0]]
        for dim in range(self.dimensions):
            for j in range(2):
                distance_to_border = np.abs(position[dim] - self.cell_positions[dim][j*-1])
                #print("cell_positions[{},{}] = {}.  position = {}.  distance_to_border = {}".format(dim, j*-1, self.cell_positions[dim][j*-1], position, distance_to_border))
                if distance_to_border < self.buffer_distance[dim]:
                    size_increase[dim][j] = self.buffer_distance[dim]
                    expand = True

        if expand:
            print('expanding map automatically with size: ' + str(size_increase))
            self._expand(size_increase)

    def _mark_visible_cells(self, center):
        # currently only works in 2D
        # TODO: generalize to 3D
        # approximate function: distance is calculated from cell perspective, not position perspective
        center_x = center[0]
        center_y = center[1]

        size_x = self.map_size[0]
        size_y = self.map_size[1]
        size_z = self.map_size[2]

        scale_x, scale_y, _ = self.cell_scale
        radius = self.vision_range

        x, y = np.ogrid[-center_x:size_x-center_x, -center_y:size_y-center_y]
        mask = (x*scale_x) ** 2 + (y*scale_y) ** 2 <= radius ** 2
        mask_z = np.repeat(mask[:, :, np.newaxis], size_z, axis=2)
        num_detected = np.sum(mask_z) - np.sum(self.cell_map['visible'][mask_z])
        self.cell_map['visible'][mask_z] = 1

        return num_detected


    def update(self, new_position, detected_obstacles=(), detected_objects=()):
        """

        :param new_position: tuple or list, (x, y, z)
        :param detected_obstacles: list of absolute positions
        :param detected_objects: list of absolute positions
        :return: updated local map
        """
        #print("current_position: {}".format(current_position))
        self._automatic_expansion(new_position)
        #print("_get_cell({}) = {}".format(current_position, self._get_cell(current_position)))
        self.position = new_position
        self._visit_cell(self._get_cell(new_position))
        num_detected = self._mark_visible_cells(self._get_cell(new_position))
        #print("value of cell {} : {}".format(self._get_cell(current_position), self.cell_map[self._get_cell(current_position)]))
        for obstacle_position in detected_obstacles:
            self._mark_cell(self._get_cell(obstacle_position), 'obstacle')
        for object_position in detected_objects:
            self._mark_cell(self._get_cell(object_position), 'object')

        return self.get_local_map(), num_detected

    def get_local_map(self):
        current_cell = self._get_cell(self.position)

        local_idx = []
        for dim in range(self.dimensions):
            start = np.max((current_cell[dim] - self.local_map_size[dim] // 2, int(0)))
            end = start + self.local_map_size[dim]
            local_idx.append(slice(start, end))
        local_idx = tuple(local_idx)
        #print("local idx: " + str(local_idx))
        local_cell_map = np.concatenate([np.clip(v[local_idx] // self.thresholds[k], 0, 1) for k, v in self.cell_map.items()], axis=2)
        #local_cell_map = {k: v[local_idx] // self.thresholds[k] for k, v in self.cell_map.items()}
        return local_cell_map  # self.cell_map[local_idx]

    def visualize(self, num_ticks_approx=10, cell_map=None):

        if cell_map is not None:
            vis_map = cell_map
        else:
            vis_map = np.concatenate([v // self.thresholds[k] for k, v in self.cell_map.items()], axis=2)

        len_z = self.local_map_size[2]
        temp_map = []
        for k in range(len(self.thresholds.keys())):
            temp_map.append(np.sum(vis_map[:, :, k*len_z:(k+1)*len_z], axis=2).clip(0, 1)*(k+1))

        vis_map = np.max(np.stack(temp_map, axis=2), axis=2)
        #print(vis_map.max())
        vis_map = vis_map.transpose()

        cmap = colors.ListedColormap(['midnightblue', 'lightsteelblue', 'limegreen', 'red', 'gold'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(vis_map, origin='lower', cmap=cmap, norm=norm)

        if cell_map is not None:
            local_cell_positions = []
            for dim in range(self.dimensions):
                start = self._get_cell(self.position)[dim] - self.local_map_size[dim] // 2
                start = np.max([start, 0])
                end = start + self.local_map_size[dim]
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
        plt.xticks(x_tick_pos[::x_tick_skip], x_tick_val[::x_tick_skip])
        plt.yticks(y_tick_pos[::y_ticks_skip], y_tick_val[::y_ticks_skip])
        plt.show()

    def get_info(self, position):
        return {k: bool(v[self._get_cell(position)]) for k, v in self.cell_map.items()}

    def get_current_position(self):
        return np.array(self.position, dtype=float)


if __name__ == '__main__':
    print('===== 0 =====')
    m = GlobalMap(map_size=(6, 6, 1), starting_position=(0, 0, 0), cell_scale=(1, 1, 1),
                  buffer_distance=(5, 5, 0), local_map_size=(5, 5, 1), vision_range=5,
                  detection_threshold_obstacle=1, detection_threshold_object=1)
    #m.visualize()
    loc, _ = m.get_local_map()
    m.visualize(cell_map=loc)
    #print("visited: " + str(np.nonzero(m.cell_map)))
    print('===== 1 =====')
    loc, _ = m.update((-1, 0, 0), [[1,0,0],[1,1,0]])
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 2 =====')
    loc, _ = m.update((-2, 0, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 3 =====')
    loc, _ = m.update((-3, 0, 0), [[-4,1,0],[-4,0,0],[-4,-1,0]])
   # m.visualize()
    m.visualize(cell_map=loc)

    print('===== 4 =====')
    loc, _ = m.update((-3, 1, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 5 =====')
    loc, _ = m.update((-3, 2, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 6 =====')
    loc, _ = m.update((-3, 3, 0), [[-3,5,0],[-2,5,0]])
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 7 =====')
    loc, _ = m.update((-4, 3, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 8 =====')
    loc, _ = m.update((-5, 3, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 9 =====')
    loc, _ = m.update((-6, 3, 0), [[-7,2,0],[-7,3,0],[-7,4,0]], detected_objects=[[-5,5,0]])
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 10 =====')
    loc, _ = m.update((-5, 3, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 11 =====')
    loc, _ = m.update((-4, 3, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 12 =====')
    loc, _ = m.update((-4, 4, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 13 =====')
    loc, _ = m.update((-3, 4, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 14 =====')
    loc, _ = m.update((-2, 4, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    print('===== 15 =====')
    loc, _ = m.update((-1, 4, 0))
    #m.visualize()
    m.visualize(cell_map=loc)

    #print("position [38,38,0]: " + str(m._get_position([38,38,0])))
    #m._expand(((5,0),(0,0),(0,0)))
    m.visualize()
    #print("visited: " + str(np.nonzero(m.update((-1, 0, 0), []))))
    #print("position [24,38,0]: " + str(m._get_position([24,38,0])))
    #m._expand(((0,0),(0,9.9),(2,0)))
    #print("value: " + str(m.cell_map[24,26,0]))
    #m.step((-1,1,0), [(-1,3,0),(-2,3,0)])
    #m.step((0,1,0),[])
