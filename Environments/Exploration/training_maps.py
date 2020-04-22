import random
import numpy as np

N_TRAINING_MAPS = 6
N_VALIDATION_MAPS = 3


# Intended settings: cell_scale = (1, 1, 1), map_size[z] = 5 (z\in [0, 4]), z_pos = 2
def generate_map(map_idx):

    X = np.linspace(0, 60, 61)
    Y = np.linspace(0, 60, 61)
    Z = np.linspace(0, 2, 3)
    z_pos = 1
    obstacles = []

    # add borders
    for z in Z[1:-1]:
        for x in X:
            obstacles.append([x, Y[0], z])
            obstacles.append([x, Y[1], z])
            obstacles.append([x, Y[-1], z])
            obstacles.append([x, Y[-2], z])

        for y in Y:
            obstacles.append([X[0], y, z])
            obstacles.append([X[-1], y, z])
            obstacles.append([X[1], y, z])
            obstacles.append([X[-2], y, z])

    # add floor and ceiling
    for x in X:
        for y in Y:
            obstacles.append([x, y, Z[0]])
            obstacles.append([x, y, Z[-1]])

    if map_idx == -1:
        map_idx = random.randint(1, N_TRAINING_MAPS)
    elif map_idx == -2:
        map_idx = random.randint(N_TRAINING_MAPS+1, N_TRAINING_MAPS+N_VALIDATION_MAPS)
    elif map_idx == -3:
        map_idx = random.randint(1, N_TRAINING_MAPS+N_VALIDATION_MAPS)

    if map_idx == 1:
        starting_position = [np.random.uniform(5, 55), np.random.uniform(5, 55)]

    if map_idx == 2:

        for z in Z[1:-1]:
            for x in X[:51]:
                obstacles.append([x, 12, z])
                obstacles.append([x, 36, z])
            for x in X[10:]:
                obstacles.append([x, 24, z])
                obstacles.append([x, 48, z])

        starting_position = random.choice([[6, 6], [54, 54]])

    if map_idx == 3:
        for z in Z[1:-1]:
            for x in X[15:46]:
                obstacles.append([x, 45, z])
            for y in Y[15: 46]:
                obstacles.append([15, y, z])
                obstacles.append([45, y, z])
            for y in Y[:31]:
                obstacles.append([30, y, z])

        starting_position = random.choice([[7, 7], [52, 52],[7, 52], [52, 7], [30, 37], [15, 7], [45, 7], [30, 52]])

    if map_idx == 4:
        for z in Z[1:-1]:
            for x in X[:26]:
                obstacles.append([x, 45, z])
            for x in X[35:]:
                obstacles.append([x, 45, z])
            for x in X[15:46]:
                obstacles.append([x, 25, z])
            for y in Y[:26]:
                obstacles.append([30, y, z])
            for y in Y[35:46]:
                obstacles.append([25, y, z])
                obstacles.append([35, y, z])

            starting_position = random.choice([[7, 7], [52, 52], [7, 52], [52, 7], [20, 30], [40, 30]])

    if map_idx == 5:
        for z in Z[1:-1]:
            for x in X[15:46]:
                obstacles.append([x, 20, z])
                obstacles.append([x, 40, z])
            for y in Y[:21]:
                obstacles.append([30, y, z])
            for y in Y[40:]:
                obstacles.append([30, y, z])

        starting_position = random.choice([[7, 7], [52, 52], [7, 52], [52, 7], [7, 30], [52, 30], [30, 30]])

    if map_idx == 6:
        for z in Z[1:-1]:
            for x in X[10:21]:
                obstacles.append([x, 20, z])
                obstacles.append([x, 40, z])
            for x in X[20:41]:
                obstacles.append([x, 30, z])
            for x in X[40:51]:
                obstacles.append([x, 20, z])
                obstacles.append([x, 40, z])
            for y in Y[10:21]:
                obstacles.append([20, y, z])
                obstacles.append([40, y, z])
            for y in Y[20:41]:
                obstacles.append([30, y, z])
            for y in Y[40:51]:
                obstacles.append([20, y, z])
                obstacles.append([40, y, z])

        starting_position = random.choice([[7, 7], [52, 52], [7, 52], [52, 7], [7, 30], [52, 30], [30, 7], [30, 52]])

    if map_idx == 7:
        for z in Z[1:-1]:
            for y in Y[:51]:
                obstacles.append([12, y, z])
                obstacles.append([36, y, z])
            for y in Y[10:]:
                obstacles.append([24, y, z])
                obstacles.append([48, y, z])

        starting_position = random.choice([[6, 6], [54, 54]])

    if map_idx == 8:
        for z in Z[1:-1]:
            for x in X[12:49]:
                obstacles.append([x, 30, z])
            for y in Y[12:49]:
                obstacles.append([30, y, z])

        starting_position = random.choice([[15, 15], [45, 45], [15, 45], [45, 15]])

    if map_idx == 9:
        for z in Z[1:-1]:
            for x in X[:11]:
                obstacles.append([x, 30, z])
            for x in X[20:41]:
                obstacles.append([x, 30, z])
            for x in X[50:]:
                obstacles.append([x, 30, z])
            for y in Y[:11]:
                obstacles.append([30, y, z])
            for y in Y[20:41]:
                obstacles.append([30, y, z])
            for y in Y[50:]:
                obstacles.append([30, y, z])

        starting_position = random.choice([[15, 15], [45, 45], [15, 45], [45, 15]])

    starting_position.append(z_pos)
    starting_position = np.array(starting_position, dtype=np.float32)
    solved_threshold = 3000

    return starting_position, obstacles, solved_threshold
