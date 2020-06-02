import argparse, json, msvcrt, airsim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Environments.env_utils import make_env_utils
# TODO: Check that this script works


class HumanController:
    def __init__(self, env, env_utils, img_idx, eval_steps):
        self.img_idx = img_idx
        self.env = env
        self.env_utils = env_utils
        self.eval_steps = eval_steps

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

    def capture_depth(self):
        requests = [airsim.ImageRequest(
            'front_center', airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False)]
        responses = self.env.client.simGetImages(requests)
        response = responses[0]
        depth = airsim.list_to_2d_float_array(
            response.image_data_float, response.width, response.height)
        plt.imshow(depth, cmap='gray', vmin=-5, vmax=30)
        plt.savefig("ObjectDetection/airsim_imgs/depth/image{}.png".format(self.img_idx), dpi=300)
        print("Image {} captured!".format(self.img_idx))
        self.img_idx += 1

    def fly(self):
        step = 0
        log_dict = {'TotalReturn': 0,
                    'nCrashes': 0,
                    'TotalSteps': 0,
                    'nTerminationsCorrect': 0,
                    'nTerminationsIncorrect': 0}
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
                step = 0
                log_dict = {'TotalReturn': 0,
                            'nCrashes': 0,
                            'TotalSteps': 0,
                            'nTerminationsCorrect': 0,
                            'nTerminationsIncorrect': 0}
                env.reset()
            elif key == 'c':
                self.capture_image()
            elif key == 'f':
                self.capture_depth()
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
                    if step == self.eval_steps:
                        print('EVALUATION FINISHED')
                        break
                step += 1

        env.close()
        print("Log dict:")
        print(log_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--image_number', type=int, default=0)
    args = parser.parse_args()
    with open(args.parameters) as f:
        parameters = json.load(f)

    eval_steps = parameters['eval']['n_eval_steps']
    env_utils, env = make_env_utils(**parameters)
    hc = HumanController(env, env_utils, args.image_number, eval_steps)
    hc.fly()
