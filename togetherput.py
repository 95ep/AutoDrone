import argparse
import json
import os
import torch
import numpy as np

from Environments.env_utils import make_env_utils
from Agents.neutral_net import NeutralNet


class AutonomousDrone:
    """
    Observation: local map, global map?
    Action: Waypoint (dx, dy)
    Reward:
    """
    def __init__(self, **parameters):
        parameters_air = {'env_str': 'AirSim'}
        parameters_air['AirSim'] = parameters['AirSim']

        parameters_exploration = {'env_str': 'Exploration'}
        parameters_exploration['Exploration'] = parameters['Exploration']

        self.env_utils_air, self.env_air = make_env_utils(**parameters_air)
        self.env_utils_exploration, self.env_exploration = make_env_utils(**parameters_exploration)

        network_kwargs = self.env_utils_air.get_network_kwargs()
        # Add additional kwargs from parameter file
        network_kwargs.update(parameters['neural_network'])

        self.point_navigator = NeutralNet(**network_kwargs)
        self.point_navigator.load_state_dict(torch.load(parameters['point_navigation']['weights']))
        self.object_detection_frequency = parameters['point_navigation']['object_detection_frequency']
        self.obstacle_detection_frequency = parameters['point_navigation']['obstacle_detection_frequency']
        self.fov_angle = parameters['Exploration']['fov_angle']
        self.dead = True
        self.reward_scaling = (self.env_exploration.cell_map.vision_range / self.env_exploration.cell_map.cell_scale[0]) * \
                              (self.env_exploration.cell_map.vision_range / self.env_exploration.cell_map.cell_scale[1]) * np.pi

    def reset(self):
        self.dead = False
        _ = self.env_air.reset()
        return self.env_exploration.reset(starting_position=self.env_air.get_position(), starting_direction=self.env_air.get_direction())

    def step(self, action):
        """

        :param action: delta_position - relative position of next waypoint - tuple-like: (dx, dy)
        :return:
        """
        if len(action) == 2:
            delta_pos = np.concatenate([np.array(action, dtype=float), np.array([0], dtype=float)])  # add z-dim
        else:
            delta_pos = np.array(action, dtype=float)
        waypoint = self.env_exploration.cell_map.position + delta_pos
        print("POSITION: ", self.env_exploration.cell_map.position)
        self.env_air.target_position = waypoint

        obs_air = self.env_air._get_state()
        done, reached_destination, collision = False, False, False
        steps = 0
        vision_reward = 0
        # move to waypoint
        while not done:
            obs_vector, obs_visual = self.env_utils_air.process_obs(obs_air)
            comb_obs = tuple(o for o in [obs_vector, obs_visual] if o is not None)
            with torch.no_grad():
                value, action, log_prob = self.point_navigator.act(comb_obs)
            action = self.env_utils_air.process_action(action)
            obs_air, reward, collision, info = self.env_air.step(action)
            if collision:
                done = True
                self.dead = True
            if action == 0:
                done = True
                reached_destination = info['terminated_at_target']

            object_positions = []
            obstacle_positions = []
            if steps % self.object_detection_frequency == 0:
                pass  # TODO: implement
            if steps % self.obstacle_detection_frequency == 0:
                obstacle_positions = self.env_air.get_obstacles(field_of_view=self.fov_angle)
            pos = self.env_air.get_position()
            direction = self.env_air.get_direction()
            _, new_vision = self.env_exploration.cell_map.update(pos.copy(), detected_objects=object_positions,
                                                                 detected_obstacles=obstacle_positions,
                                                                 camera_direction=direction)
            vision_reward += new_vision
            steps += 1

        reward = vision_reward / self.reward_scaling + 5 * (reached_destination - 1) - 0.01 * steps
        return self.env_exploration.cell_map.get_local_map(), reward, collision, {}

    def render(self, local=True):
        self.env_exploration.render(local)

    def close(self):
        pass
