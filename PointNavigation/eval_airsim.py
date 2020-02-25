import json
import torch
import airgym
from risenet.neutral_net import NeutralNet
from trainer_new import process_obs

# Should only need to modify here
weights_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/neutral-airsim/saved_models/model260.pth"
parameters_path = "D:/Exjobb2020ErikFilip/AutoDrone/runs/neutral-airsim/parameters.json"
n_eval_steps = 128

# Read paraneters
with open(parameters_path) as f:
    parameters = json.load(f)

env_str = 'AirSim'
env = airgym.make(sensors=parameters['environment']['sensors'], max_dist=parameters['environment']['max_dist'],
                    REWARD_SUCCESS=parameters['environment']['REWARD_SUCCESS'],
                    REWARD_FAILURE=parameters['environment']['REWARD_FAILURE'],
                    REWARD_COLLISION=parameters['environment']['REWARD_COLLISION'],
                    REWARD_MOVE_TOWARDS_GOAL=parameters['environment']['REWARD_MOVE_TOWARDS_GOAL'],
                    REWARD_ROTATE=parameters['environment']['REWARD_ROTATE'],
                    height=parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height'],
                    width=parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width'])

vector_encoder, visual_encoder, compass_encoder = False, False, False
vector_shape, visual_shape, compass_shape = None, None, None
n_actions = env.action_space.n

if 'rgb' in parameters['environment']['sensors']:
    visual_encoder = True
    h = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
    w = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
    visual_shape = (h, w, 3)

if 'depth' in parameters['environment']['sensors']:
    if visual_encoder:
        visual_shape = (visual_shape[0], visual_shape[1], visual_shape[2]+1)
    else:
        visual_encoder = True
        h = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Height']
        w = parameters['airsim']['CameraDefaults']['CaptureSettings'][0]['Width']
        visual_shape = (h, w, 1)

if 'pointgoal_with_gps_compass' in parameters['environment']['sensors']:
    vector_encoder = True
    vector_shape = (3,)

ac = NeutralNet(has_vector_encoder=vector_encoder, vector_input_shape=vector_shape,
                has_visual_encoder=visual_encoder, visual_input_shape=visual_shape,
                has_compass_encoder=compass_encoder, compass_input_shape=compass_shape,
                num_actions=n_actions, has_previous_action_encoder=False,
                hidden_size=32, num_hidden_layers=2)

ac.load_state_dict(torch.load(weights_path))

obs = env.reset()
obs_vector, obs_visual, obs_compass = process_obs(obs, env_str, parameters)
comb_obs = tuple(o for o in [obs_vector, obs_visual, obs_compass] if o is not None)

total_eval_ret = 0
n_collisions = 0
n_terminate_correct = 0
n_terminate_incorrect = 0
done = False
for step in range(n_eval_steps): # Use n_eval as steps to evaluate in
    with torch.no_grad():
        value, action, _ = ac.act(comb_obs, deterministic=True)
    next_obs, reward, done, info = env.step(action.item())
    if 'terminated_at_target' in info:
        if info['terminated_at_target']:
            n_terminate_correct += 1
        else:
            n_terminate_incorrect += 1
    total_eval_ret += reward
    obs_vector, obs_visual, obs_compass = process_obs(next_obs, env_str, parameters)
    comb_obs = tuple(o for o in [obs_vector, obs_visual, obs_compass] if o is not None)
    if done:
        obs = env.reset()
        obs_vector, obs_visual, obs_compass = process_obs(obs, env_str, parameters)
        comb_obs = tuple(o for o in [obs_vector, obs_visual, obs_compass] if o is not None)
        n_collisions += 1

print("Evaluated for {} steps. Total return was {} and number of collisions {}".format(n_eval_steps, total_eval_ret, n_collisions))
print("{} terminations were correct and {} were incorrect".format(n_terminate_correct, n_terminate_incorrect))
