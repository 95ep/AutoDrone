import numpy as np
import json
from togetherput import AutonomousDrone

with open('Parameters/parameters_ad.json') as f:
    parameters = json.load(f)

ad = AutonomousDrone(**parameters)

obs = ad.reset()
ad.render(False)
action = np.array([-1.,0])
obs, rew, done, info = ad.step(action)

ad.env_exploration.cell_map.visualize3d()
