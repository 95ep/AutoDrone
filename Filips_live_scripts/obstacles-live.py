from Environments.env_utils import make_env_utils
import json
import matplotlib.pyplot as plt
import numpy as np

with open("D:/Exjobb2020ErikFilip/AutoDrone/Parameters/parameters_airsim.json") as f:
    parameters = json.load(f)

env_utils, env = make_env_utils(**parameters)

all_obstac = []
obstacles = env.get_obstacles(1.57, 64)
for o in obstacles:
   all_obstac.append(o)

fig = plt.figure()
ax = fig.gca(projection='3d')

array = np.array(all_obstac)
ax.scatter(array[:,0], array[:,1], array[:,2])
plt.show()
