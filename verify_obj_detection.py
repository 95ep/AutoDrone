# All positions of monitors
gt_monitor_positions = [
[12.3, -4.64, 0.05],
[12.35, -3.22, 0.05],
[12.29, 5.45, 0.05],
[12.3, 3.88, 0.05],
[12.3, 2.44, 0.05],
[12.7, -3.01, 0.05],
[12.67, -4.47, 0.05],
[12.7, 2.65, 0.05],
[12.7, 4.1, 0.05],
[12.7, 5.55, 0.05],
[18.44, -4.64, 0.05],
[18.45, -3.22, 0.05],
[18.43, 5.45, 0.05],
[18.44, 3.88, 0.05],
[18.44, 2.44, 0.05],
[18.84, -3.01, 0.05],
[18.81, -4.47, 0.05],
[18.84, 2.65, 0.05],
[18.84, 4.1, 0.05],
[18.84, 5.55, 0.05],
[6.42, -4.64, 0.05],
[6.42, -3.22, 0.05],
[6.42, 5.45, 0.05],
[6.42, 3.88, 0.05],
[6.42, 2.44, 0.05],
[6.82, -3.01, 0.05],
[6.79, -4.47, 0.05],
[6.82, 2.65, 0.05],
[6.82, 4.1, 0.05],
[6.82, 5.55, 0.05],
[0.47, -4.64, 0.05],
[0.47, -3.22, 0.05],
[0.47, 5.45, 0.05],
[0.47, 3.88, 0.05],
[0.47, 2.44, 0.05],
[0.87, -3.01, 0.05],
[0.87, -4.47, 0.05],
[0.87, 2.65, 0.05],
[0.87, 4.1, 0.05],
[0.87, 5.55, 0.05]
]

def precision_recall(object_positions, gt_monitor_positions, scale):
    x_scale = scale[0]
    y_scale = scale[1]
    z_scale = scale[2]
    # Calculate precision
    n_true_positive = 0
    n_false_positive = 0
    for pos in object_positions:
        correct_pos = False
        for i, gt_pos in enumerate(gt_monitor_positions):
            if abs(pos[0] - gt_pos[0]) < (x_scale) and \
                    abs(pos[1] - gt_pos[1]) < (y_scale) and \
                    abs(pos[2] - gt_pos[2]) < (z_scale):
                correct_pos = True
                n_true_positive += 1
                break
        if not correct_pos:
            n_false_positive += 1

    precision = n_true_positive / len(object_positions)

    n_recalled = 0
    for gt_pos in gt_monitor_positions:
        for pos in object_positions:
            if abs(pos[0] - gt_pos[0]) < (x_scale) and \
                    abs(pos[1] - gt_pos[1]) < (y_scale) and \
                    abs(pos[2] - gt_pos[2]) < (z_scale):
                n_recalled += 1
                break

    recall = n_recalled / len(gt_monitor_positions)

    print("n true postive {}".format(n_true_positive))
    print("n false positive {}".format(n_false_positive))
    print("n monitors found {}".format(n_recalled))
    print("Precision {}, recall {}".format(precision, recall))
    print("All positions of objects detected")
    print(object_positions)

    return precision, recall


if __name__ == '__main__':
    import numpy as np
    from Environments.Exploration.map_env import make, AirSimMapEnv
    import json
    import argparse
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)

    kwargs = parameters["Exploration"]
    m = AirSimMapEnv(**kwargs, **parameters)
    x_scale = kwargs['cell_scale'][0]
    y_scale = kwargs['cell_scale'][1]
    z_scale = kwargs['cell_scale'][2]

    obs = m.reset()

    # Make the moves
    waypoints = [
    np.array([-0.5, -3.5]),
    np.array([4, -3.5]),
    np.array([10, -3.5]),
    np.array([15, -3.5]),
    np.array([20.5, -3.5]),
    np.array([15, -3.5]),
    np.array([10, -3.5]),
    np.array([-0.5, -3.5]),
    np.array([-0.5, 4.0]),
    np.array([4.5, 4.0]),
    np.array([10.5, 4.0]),
    np.array([20, 4.0]),
    np.array([10, 4.0]),
    np.array([5, 4.0]),
    np.array([-0.5, 4.5]),
    ]

    for goal in waypoints:
        print("Goal: {}".format(goal))
        not_reached = True
        pos = m.env_airsim.get_position()
        delta =  np.array([goal[0]-pos[0], goal[1]-pos[1]])
        while not_reached:
            try:
                obs, reward, done, info = m.step(delta)
            except Exception as e:
                print(e)
                break
            pos = m.env_airsim.get_position()
            delta = np.array([goal[0]-pos[0], goal[1]-pos[1]])
            if np.sqrt(delta[0]**2 + delta[1]**2) < 0.5:
                not_reached = False
            else:
                print("trgt not reached, tries again!")
                print(pos)


    map = m._get_map(local=False, binary=False)
    # Extract objects
    obj_map = map == m.tokens['object']
    obj_cells = np.argwhere(obj_map)
    object_positions = []
    for cell in obj_cells:
        object_positions.append(m._get_position(cell))

    # Extract obstacles
    obs_map = map == m.tokens['obstacle']
    obs_cells = np.argwhere(obs_map)
    obstacle_positions = []
    for cell in obs_cells:
         obstacle_positions.append(m._get_position(cell))

    # Extract visited
    visited_map = map == m.tokens['visited']
    visited_cells = np.argwhere(visited_map)
    visited_positions = []
    for cell in visited_cells:
         visited_positions.append(m._get_position(cell))


    if len(object_positions) > 0:
        precision_recall(object_positions, gt_monitor_positions, [x_scale, y_scale, z_scale])


    save_dir = os.path.dirname(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(save_dir + '/objects.npy', np.array(object_positions))
    np.save(save_dir + '/obstacles.npy', np.array(obstacle_positions))
    np.save(save_dir + '/visited.npy', np.array(visited_positions))
    np.save(save_dir + '/ground_truth', np.array(gt_monitor_positions))

    m.render(render_3d=True)
