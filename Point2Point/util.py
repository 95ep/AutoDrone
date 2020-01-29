import numpy as np
import airsim


def printInfo(currentClient):
    pos = currentClient.simGetGroundTruthKinematics().position
    q = currentClient.simGetGroundTruthKinematics().orientation
    yawDeg = airsim.to_eularian_angles(q)[2]/np.pi*180

    print('Current yaw is {} deg'.format(yawDeg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))