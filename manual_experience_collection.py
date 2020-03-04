import argparse
import json
import pickle
import msvcrt
import torch
from Environments.env_utils import make_env_utils


class ManualCollector:

    def __init__(self, steps_per_epoch, env, env_utils, data_count):
        self.data_count = data_count
        self.steps_per_epoch = steps_per_epoch
        self.env = env
        self.env_utils = env_utils
        self.buffer = self.env_utils.make_buffer(steps_per_epoch, 0.99, 0.97)

    def save_data(self, data):
        with open('data_' + str(self.data_count) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
        self.data_count += 1

# Load
#with open('objs.pkl', 'rb') as f:
#    obj0, obj1, obj2 = pickle.load(f)

    def collect_trajectory(self):
        print('STARTING A NEW {} STEP TRAJECTORY'.format(self.steps_per_epoch))
        step = 0
        self.env.reset()
        while True:
            key = msvcrt.getwch()
            action = None
            if key == 'w':
                action = 1
            elif key == 'a':
                action = 2
            elif key == 'd':
                action = 3
            elif key == 'z':
                action = 5
            elif key == 'x':
                action = 4
            elif key == 's':
                action = 0
            elif key == 'r':
                print("ENVIRONMENT RESET BY PLAYER")
                self.env.reset()
            elif key == 'b' or key == 't':
                print('TRAJECTORY TERMINATED BY PLAYER')
                break

            if action is not None:
                obs, reward, done, info = self.env.step(action)
                value = torch.tensor([20], dtype=torch.float32)
                log_prob = torch.tensor([0], dtype=torch.float32)
                obs_vector, obs_visual = self.env_utils.process_obs(obs)
                self.buffer.store(obs_vector, obs_visual, action, reward, value, log_prob)
                step += 1
                if step == self.steps_per_epoch:
                    print('TRAJECTORY FINISHED')
                    self.buffer.finish_path(value)
                    data = self.buffer.get()
                    self.save_data(data)
                    break

        print('COLLECT NEW TRAJECTORY? y = yes, n = no')
        key = msvcrt.getch()
        if key == 'y':
            self.data_count += 1
            self.collect_trajectory()
        else:
            return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--file_number', type=int, default=0)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)

    env_utils, env = make_env_utils(**parameters)
    mc = ManualCollector(parameters['training']['steps_per_epoch'], env, env_utils, args.file_number)
    mc.collect_trajectory()
