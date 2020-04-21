import numpy as np
import vtkplotter as vtkp
from matplotlib.colors import ListedColormap

ceiling_level = -1.91

gt_pos = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/ground_truth.npy')
obstac = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/obstacles.npy')
objs = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/objects.npy')
visited = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/visited.npy')

cell_scale = np.array([0.5, 0.5, 0.5])

print("Shape of gt_pos {}".format(gt_pos.shape))
print("Shape of obstac {}".format(obstac.shape))
print("Shape of objs {}".format(objs.shape))
print("Shape of visited {}".format(visited.shape))

# Find max value for x, y, z
max_vals = np.amax(obstac, axis=0)
min_vals = np.amin(obstac, axis=0)
grid_shape = ((max_vals - min_vals) // cell_scale + 1).astype(int)
print(grid_shape)
grid = np.zeros(grid_shape)


# Add obstacles
for i in range(obstac.shape[0]):
    pos = obstac[i, :]
    if ceiling_level is not None:
        if abs(pos[2]-ceiling_level) < cell_scale[2]:
            continue
    idx = ((pos - min_vals) // cell_scale).astype(int)
    grid[idx[0], idx[1], idx[2]] = 4

factor = 2
new_grid = np.zeros((grid_shape[0]*factor, grid_shape[1]*factor, grid_shape[2]*factor))
for i in range(factor):
    for j in range(factor):
        for k in range(factor):
            new_grid[i::factor, j::factor, k::factor] = grid
grid = new_grid

print(grid.shape)
print(np.amax(grid))
print(np.amin(grid))
color_list = np.array([[0., 0., 0.3, 0.1],  # unknown starts from 1
                       [0.9, 0.95, 1., 0.2],  # visible
                       [0.2, 1., 0.2, 0.8],  # visited
                       [1., 0.2, 0.2, 0.6],  # obstacle
                       [1., 1., 0, 1.],  # object
                       [0., 0.7, 0., 1.]])  # position
newcmp = ListedColormap(color_list)

vol = vtkp.Volume(grid)
lego = vol.legosurface(vmin=2, vmax=6)
vtkp.show(lego, axes=8)
