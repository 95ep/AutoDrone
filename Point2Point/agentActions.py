import numpy as np
import airsim



def rotateLeft(currentClient):
    # Rotate UAV approx 30 deg to the left (counter-clockwise)
    currentClient.rotateByYawRateAsync(-30, 1).join()
    # Stop rotation
    currentClient.rotateByYawRateAsync(0, 1e-6).join()


def rotateRight(currentClient):
    # Rotate UAV approx 30 deg to the right (clockwise)
    currentClient.rotateByYawRateAsync(30, 1).join()
    # Stop rotation
    currentClient.rotateByYawRateAsync(0, 1e-6).join()


def moveForward(currentClient):
    q = currentClient.simGetGroundTruthKinematics().orientation
    yaw = airsim.to_eularian_angles(q)[2]
    # Calc velocity vector of magnitude 1 in direction of UAV
    vel = (np.cos(yaw), np.sin(yaw), 0) # Keep z fixed for now
    # Move forward approx 1 meter
    currentClient.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=1).join()
    # Stop the UAV
    currentClient.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()