import random
import numpy as np

N_TRAINING_MAPS = 2
N_VALIDATION_MAPS = 2

# Intended settings: cell_scale = (1, 1, 1), map_size[z] = 5 (z\in [0, 4]), z_pos = 2
def generate_map(map_idx):

    x, y, z = np.ogrid[-30:31, -30:31, 0:5]

    if map_idx == -1:
        map_idx = random.randint(1, N_TRAINING_MAPS)
    elif map_idx == -2:
        map_idx = random.randint(N_TRAINING_MAPS+1, N_TRAINING_MAPS+N_VALIDATION_MAPS)
    elif map_idx == -3:
        map_idx = random.randint(1, N_TRAINING_MAPS+N_VALIDATION_MAPS)

    if map_idx == 1:



    return starting_position, obstacles
