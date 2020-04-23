import sys
sys.path.append("C:/Users/Filip/Projects/RISE/AutoDrone")

from Environments.Exploration import map_env
import numpy as np

X = np.linspace(-10, 10, 81)
Y = np.linspace(-10, 10, 81)
Z = np.linspace(0, 10, 41)

obstacles = []

for x in X:
    for y in Y:
        obstacles.append([x, y, Z[0]])
for z in Z[1:]:
    for x in X:
        #obstacles.append([x, Y[0], z])
        obstacles.append([x, Y[-1], z])

    for y in Y:
        obstacles.append([X[0], y, z])
        obstacles.append([X[-1], y, z])

    for y in Y[int(len(Y)/2):]:
        obstacles.append([4, y, z])

for z in Z[int(len(Z) * 0.55):int(len(Z)*0.75)]:
    for y in Y[int(len(Y)*0.35):int(len(Y)*0.45)]:
        obstacles.append([2, y, z])

for z in Z[1:int(len(Z)*0.5)]:
    for y in Y[int(len(Y)*0.3):int(len(Y)*0.6)]:
        obstacles.append([7, y, z])

# Plot only cone
parameters = {"cell_scale": (0.25, 0.25, 0.25), "starting_map_size": (10, 10, 10), "local_map_dim": (1,1,1),
              "buffer_distance": (10, 10, 0), "vision_range": 8}

parameters["map_idx"] = 0

env = map_env.MapEnv(**parameters)
_ = env.reset(starting_position=[0, 0, 5])
_ = env._update([0.01, 0, 5], detected_obstacles=obstacles)
#_ = env.step([0.11, 0])
env.render(render_3d=True, voxels=True, show_detected=True)
