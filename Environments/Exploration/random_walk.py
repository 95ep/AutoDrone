from Environments.Exploration import map_env
import numpy as np


"""
sigma = 10
max_steps = 30
metrics = ['detected', 'reward']

parameters = {"cell_scale": (1, 1, 1), "starting_map_size": (10, 10, 3), "local_map_dim": (1,1,1), "buffer_distance": (10, 10, 0), "vision_range": 8}
parameters["map_idx"] = 6
parameters["interactive_plot"] = True

env = map_env.MapEnv(**parameters)
_ = env.reset()
temp_log = np.zeros((max_steps, len(metrics)))
for k in range(max_steps):
    rel_pos = np.random.multivariate_normal([0.0, 0.0], sigma * np.eye(2))
    _, reward, _, _ = env.step(rel_pos, safe_mode=True)
    total_detected = env.total_detected
    temp_log[k] = [total_detected, reward]
    env.render(render_3d=False, show_detected=True, ceiling_z=2, floor_z=0)
"""

sigmas = np.array([0.1, 1.0, 5.0])
num_maps = 3
runs_per_map = 2
max_steps = 50
metrics = ['detected', 'reward']

log = np.zeros((len(sigmas), num_maps, runs_per_map, max_steps, len(metrics)))

parameters = {"cell_scale": (1, 1, 1), "starting_map_size": (10, 10, 3), "local_map_dim": (1,1,1), "buffer_distance": (10, 10, 0), "vision_range": 8}
parameters["interactive_plot"] = True

for s in range(len(sigmas)):
    for i in range(num_maps):
        parameters["map_idx"] = i+1
        env = map_env.MapEnv(**parameters)

        for j in range(runs_per_map):
            _ = env.reset()

            temp_log = np.zeros((max_steps, len(metrics)))
            for k in range(max_steps):
                rel_pos = np.random.multivariate_normal([0.0, 0.0], sigmas[s] * np.eye(2))
                _, reward, _, _ = env.step(rel_pos, safe_mode=True)
                total_detected = env.total_detected
                temp_log[k] = [total_detected, reward]
                env.render(render_3d=False, show_detected=True, ceiling_z=2, floor_z=0)

            env.close()
            log[s, i, j] = temp_log

print(log)

if __name__ == '__main__':

    sigmas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
                       5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    num_maps = 9
    runs_per_map = 10
    max_steps = 250
    metrics = ['detected', 'reward']

    log = np.zeros((len(sigmas), num_maps, runs_per_map, max_steps, len(metrics)))

    parameters = {"cell_scale": (1, 1, 1), "starting_map_size": (10, 10, 3), "local_map_dim": (1,1,1),
                  "buffer_distance": (10, 10, 0), "vision_range": 8}

    for sigma in range(len(sigmas)):
        for i in range(num_maps):
            parameters["map_idx"] = i+1
            env = map_env.MapEnv(**parameters)

            for j in range(runs_per_map):
                _ = env.reset()

                temp_log = np.zeros((max_steps, len(metrics)))
                for k in range(max_steps):
                    rel_pos = np.random.multivariate_normal([0.0, 0.0], sigma * np.eye(2))
                    _, reward, _, _ = env.step(rel_pos, safe_mode=True)
                    total_detected = env.total_detected
                    temp_log[k] = [total_detected, reward]

                log[s, i, j] = temp_log

    print(log)
