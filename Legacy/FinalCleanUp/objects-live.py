from Environments.env_utils import make_env_utils
import json
import matplotlib.pyplot as plt
import numpy as np

with open("D:/Exjobb2020ErikFilip/AutoDrone/Parameters/parameters_collect_imgs.json") as f:
    parameters = json.load(f)

env_utils, env = make_env_utils(**parameters)

query_paths = ["D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_100.jpg",
                "D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_101.jpg"]

env.setup_object_detection(query_paths, rejection_factor=0.8, min_match_thres=10)



all_objs = []
objs = env.get_trgt_objects()
for o in objs:
   all_objs.append(o)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(-1,10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

array = np.array(all_objs)
ax.scatter(array[:,0], array[:,1], array[:,2])
plt.show()
