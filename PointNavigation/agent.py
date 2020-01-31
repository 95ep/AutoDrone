import airsim
import numpy as np


# Actions
def rotateLeft(currentClient):
    # Rotate UAV approx 10 deg to the left (counter-clockwise)
    currentClient.rotateByYawRateAsync(-20, 0.5).join()
    # Stop rotation
    currentClient.rotateByYawRateAsync(0, 1e-6).join()


def rotateRight(currentClient):
    # Rotate UAV approx 10 deg to the right (clockwise)
    currentClient.rotateByYawRateAsync(20, 0.5).join()
    # Stop rotation
    currentClient.rotateByYawRateAsync(0, 1e-6).join()


def moveForward(currentClient):
    q = currentClient.simGetGroundTruthKinematics().orientation
    yaw = airsim.to_eularian_angles(q)[2]
    # Calc velocity vector of magnitude 0.5 in direction of UAV
    vel = (0.5*np.cos(yaw), 0.5*np.sin(yaw), 0)
    # Move forward approx 0.25 meter
    currentClient.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=0.5).join()
    # Stop the UAV
    currentClient.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()


def moveUp(currentClient):
    # Move up approx 0.25 m. Note direction of z-axis.
    currentClient.moveByVelocityAsync(0, 0, -0.5, duration=0.5).join()
    # Stop the UAV
    currentClient.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()


def moveDown(currentClient):
    # Move down approx 0.25 m. Note direction of z-axis.
    currentClient.moveByVelocityAsync(0, 0, 0.5, duration=0.5).join()
    # Stop the UAV
    currentClient.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()


# Observations
def getImages(currentClient, maxDist = 10):
    # Returns one rbg image and one depth mao
    responses = currentClient.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, pixels_as_float = False, compress = False),
        airsim.ImageRequest('front_center', airsim.ImageType.DepthPerspective, pixels_as_float = True, compress=False)])

    # Convert to uint and reshape to matrix with 3 color channels
    bgr = np.reshape(airsim.string_to_uint8_array(responses[0].image_data_uint8), (responses[0].height, responses[0].width, 3))
    # Move color channels around and change representation to float in range [0, 1]
    rgb = np.array(bgr[:,:,[2,1,0]], dtype=np.float32)/255

    # Convert to 2D numpy array
    depthPerspective = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)
    # Clip array and put in range [0,1]
    depthPerspective = np.clip(depthPerspective, None, maxDist)

    return rgb, depthPerspective
