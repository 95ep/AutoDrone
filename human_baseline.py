import argparse
import json
import msvcrt
from Environments.env_utils import make_env_utils


parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str)
args = parser.parse_args()
with open(args.parameters) as f:
    parameters = json.load(f)

eval_steps = parameters['eval']['n_eval_steps']
env_utils, env = make_env_utils(**parameters)

step = 0
log_dict = {'TotalReturn': 0,
            'nCrashes': 0,
            'TotalSteps': 0,
            'nTerminationsCorrect': 0,
            'nTerminationsIncorrect': 0}
env.reset()
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
        print("Human evaluation restarted")
        step = 0
        log_dict = {'TotalReturn': 0,
                    'nCrashes': 0,
                    'TotalSteps': 0,
                    'nTerminationsCorrect': 0,
                    'nTerminationsIncorrect': 0}
        env.reset()
    elif key == 'b' or key == 't':
        print('TRAJECTORY TERMINATED BY PLAYER')
        break

    if action is not None:
        _, reward, done, info = env.step(action)
        env.render()
        log_dict['TotalSteps'] += 1
        log_dict['TotalReturn'] += reward
        if done:
            env.reset()
            log_dict['nCrashes'] += 1
        else:
            if 'terminated_at_target' in info:
                if info['terminated_at_target']:
                    log_dict['nTerminationsCorrect'] += 1
                else:
                    log_dict['nTerminationsIncorrect'] += 1

            print('\rStep: {}'.format(step), end='\r')
            if step == eval_steps:
                print('EVALUATION FINISHED')
                break
        step += 1

env.close()
print("Log dict")
print(log_dict)
