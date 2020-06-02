import os
import argparse
import airsim
import json
import msvcrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
from Environments.airgym import agent_controller
from Environments.airgym.utils import reset

class CollisionChecker():

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()


    def manual_flight(self):
        print("CollisionInfo by pressing 'c', reset by pressing r and quit by pressing 't'")
        reset(self.client)
        while True:
            key = msvcrt.getwch()
            action = None
            if key == 'w':
                agent_controller.move_forward(self.client)
            elif key == 'a':
                agent_controller.rotate_left(self.client)
            elif key == 'd':
                agent_controller.rotate_right(self.client)
            elif key == 'z':
                agent_controller.move_down(self.client)
            elif key == 'x':
                agent_controller.move_up(self.client)

            elif key == 'r':
                print("ENVIRONMENT RESET BY PLAYER")
                reset(self.client, "basic23")
            elif key == 'b' or key == 't':
                print('TERMINATED BY PLAYER')
                break
            elif key == 'c':
                print(self.client.simGetCollisionInfo())
            elif key == 'p':
                print(self.client.simGetGroundTruthKinematics().position)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)


    cc = CollisionChecker()
    cc.manual_flight()
