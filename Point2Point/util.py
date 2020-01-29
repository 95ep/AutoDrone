import numpy as np


def quaternion2Yaw(q):
    # Convert from quaternion to Yaw in radians (according to Wikipedia)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def printInfo(currentClient):
    pos = currentClient.simGetGroundTruthKinematics().position
    q = currentClient.simGetGroundTruthKinematics().orientation
    yawDeg = quaternion2Yaw(q)/np.pi*180

    print('Current yaw is {} deg'.format(yawDeg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))