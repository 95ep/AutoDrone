import airsim
import cv2
import os
import numpy as np


def getImages(currentClient):
    responses = currentClient.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
        airsim.ImageRequest('front_center'), airsim.ImageType.DepthPlanner, True, False])

    return responses

def saveImages(im):
    pass


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

airsim.wait_key('Press any key capture images')

responses = getImages(client)
dir_path = os.path.dirname(os.path.realpath(__file__))


for idx, response in enumerate(responses):

    filename = os.path.join(dir_path, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png