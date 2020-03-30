import os
import argparse
import airsim
import json
import msvcrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
from Environments.env_utils import make_env_utils


class ObjectCollector:

    def __init__(self, env, env_utils, q_paths):
        self.env = env
        self.env_utils = env_utils
        self.env.setup_object_detection(q_paths, rejection_factor=0.8, min_match_thres=10)

    def capture_object(self):
        point_cloud, dst_list = self.env.get_trgt_objects()
        if len(dst_list) > 0:
            requests = [airsim.ImageRequest(
                'front_center', airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
            responses = self.env.client.simGetImages(requests)
            response = responses[0]
            bgr = np.reshape(airsim.string_to_uint8_array(
                response.image_data_uint8), (response.height, response.width, 3))
            rgb = np.array(bgr[:, :, [2, 1, 0]])

            train_image = rgb.astype(np.uint8)
            train_image = cv.cvtColor(train_image, cv.COLOR_BGR2GRAY)
            for dst in dst_list:
                img_boxes = cv.polylines(train_image,[np.int32(dst)],True,255,3, cv.LINE_AA)
            plt.imshow(train_image, cmap='gray'), plt.show()
        else:
            print("No objects found")

    def collect_objects(self):
        print("Capture by pressing 'c', reset by pressing r and quit by pressing 't'")
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
                print('TERMINATED BY PLAYER')
                break
            elif key == 'c':
                self.capture_object()

            if action is not None:
                obs, reward, done, info = self.env.step(action)
                if done:
                    self.env.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)

    q_paths = ["D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_100.jpg"]#,
                    #"D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_101.jpg"]
    env_utils, env = make_env_utils(**parameters)
    oc = ObjectCollector(env, env_utils, q_paths)
    oc.collect_objects()
