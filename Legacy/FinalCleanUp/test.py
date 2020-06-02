import numpy as np
from Environments.Exploration.map_env import make, AirSimMapEnv
import time
import json

with open('D:/Exjobb2020ErikFilip/AutoDrone/Parameters/parameters_ad.json') as f:
    parameters = json.load(f)

#m = make(vision_range=1, interactive_plot=False, local_map_dim=(16, 16, 1))
kwargs = parameters['Exploration']
m = AirSimMapEnv(**kwargs, **parameters)

obs = m.reset()
obs, reward, done, info = m.step(np.array([-1., 0.]))
m.render()
print(reward)
time.sleep(3)
token_map = m._get_map(binary=False)


#
import numpy as np
from Environments.Exploration.map_env import make, AirSimMapEnv
m = make(starting_map_size=(40,40,2), cell_scale=(0.5,0.5,0.5), vision_range=20)
_ = m.reset()
_ = m._update(np.array([0,0,0]), orientation=np.array([1,0]), detected_obstacles=([10,5,-5],[3,0,0],[3,0,0.5],[3,0.5,0],[3,0.5,0.5],[2,1,1],[3,-1,-1]))
_ = m.step(np.array([0.1,0]))
m.render(render_3d=True, show_detected=True)
