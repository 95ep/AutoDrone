import os
import argparse
import airsim
import json
import msvcrt
import numpy as np
from PIL import Image
from Environments.env_utils import make_env_utils


class ImageCollector:

    def __init__(self, env, env_utils, img_idx):
        self.img_idx = img_idx
        self.env = env
        self.env_utils = env_utils

    def capture_image(self):
        requests = [airsim.ImageRequest(
            'front_center', airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
        responses = self.env.client.simGetImages(requests)
        response = responses[0]
        bgr = np.reshape(airsim.string_to_uint8_array(
            response.image_data_uint8), (response.height, response.width, 3))
        rgb = np.array(bgr[:, :, [2, 1, 0]])
        img = Image.fromarray(rgb)
        img.save("ObjectDetection/airsim_imgs/image{}.jpg".format(self.img_idx))
        print("Image {} captured!".format(self.img_idx))
        self.img_idx += 1

    def collect_images(self):
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
                self.capture_image()

            if action is not None:
                obs, reward, done, info = self.env.step(action)
                if done:
                    self.env.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--image_number', type=int, default=0)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)

    env_utils, env = make_env_utils(**parameters)
    ic = ImageCollector(env, env_utils, args.image_number)
    ic.collect_images()
