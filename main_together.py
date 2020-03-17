import numpy as np
import json
from togetherput import AutonomousDrone

with open('Parameters/parameters_ad.json') as f:
    parameters = json.load(f)

ad = AutonomousDrone(**parameters)

ad.reset()
ad.render(False)
action = np.array([-5.,0])
ad.step(action)
