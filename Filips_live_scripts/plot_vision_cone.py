import sys
sys.path.append("C:/Users/Filip/Projects/RISE/AutoDrone")

from Environments.Exploration import map_env
import numpy as np

X = np.linspace(-10, 10, 101)
Y = np.linspace(-10, 10, 101)
Z = np.linspace(0, 10, 76)

obstacles = []

for x in X:
    for y in Y:
        obstacles.append([x, y, Z[0]])
for z in Z[1:]:
    for x in X:
        obstacles.append([x, Y[0], z])
        obstacles.append([x, Y[-1], z])

    for y in Y:
        obstacles.append([X[0], y, z])
        obstacles.append([X[-1], y, z])

    for y in Y[100:]:
        obstacles.append([2, y, z])

for z in Z[35:45]:
    for y in Y[70:90]:
        obstacles.append([4, y, z])

for z in Z[55:65]:
    for y in Y[40:60]:
        obstacles.append([7, y, z])

# Plot only cone
parameters = {"cell_scale": (0.2, 0.2, 0.2), "starting_map_size": (20, 20, 20), "local_map_dim": (1,1,1),
              "buffer_distance": (10, 10, 0), "vision_range": 6}

parameters["map_idx"] = 0

env = map_env.MapEnv(**parameters)
_ = env.reset(starting_position=[0, 0, 5])
_ = env._update([0.01, 0, 5], detected_obstacles=obstacles)
_ = env.step([0.11, 0])
env.render(render_3d=True, voxels=False, show_detected=True)
