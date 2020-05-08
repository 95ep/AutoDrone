import numpy as np
import vtkplotter as vtkp
from matplotlib.colors import ListedColormap

ceiling_level = -2.0
# ceiling_level = None

save_pth = '/Users/erikpersson/PycharmProjects/AutoDrone/plots/obstac_ver.png'
gt_pos = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_obstac/ground_truth.npy')
obstac = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_obstac/obstacles.npy')
objs = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_obstac/objects.npy')
visited = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_obstac/visited.npy')

cell_scale = np.array([0.71, 0.71, 0.71])

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

# Add visited
# for i in range(visited.shape[0]):
#     pos = visited[i, :]
#     idx = ((pos - min_vals) // cell_scale).astype(int)
#     grid[idx[0], idx[1], idx[2]] = 3

# Add obstacles
for i in range(obstac.shape[0]):
    pos = obstac[i, :]
    if ceiling_level is not None:
        if abs(pos[2]-ceiling_level) < cell_scale[2]:
            continue
    idx = ((pos - min_vals) // cell_scale).astype(int)
    grid[idx[0], idx[1], idx[2]] = 4

factor = 4
new_grid = np.zeros((grid_shape[0]*factor, grid_shape[1]*factor, grid_shape[2]*factor))
for i in range(factor):
    for j in range(factor):
        for k in range(factor):
            new_grid[i::factor, j::factor, k::factor] = grid
grid = new_grid

color_list = np.array([[0.9, 0.95, 1., 0.2], [0.2, 1., 0.2, 0.8], [1., 0.0, 0.0, 0.1], [1., 1., 0, 1.], [0., 0.7, 0., 1.]])

newcmp = ListedColormap(color_list)

vol = vtkp.Volume(grid)
lego = vol.legosurface(vmin=2, vmax=6, cmap=newcmp)
vtkp.show(lego, axes=0, azimuth=70, elevation=165, roll=-90, interactive=False)
vtkp.screenshot(save_pth)
