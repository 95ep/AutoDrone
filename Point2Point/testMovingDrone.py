import airsim
import numpy as np
import time


def quaternion2Yaw(q):
    # Convert from quaternion to Yaw (according to Wikipedia)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp) /(2*np.pi)*360
    return yaw


def rotateLeft(client):
    # Rotate UAV approx 30 deg to the left (counter-clockwise)
    client.rotateByYawRateAsync(-30, 1).join()
    # Stop rotation
    client.rotateByYawRateAsync(0, 1e-6).join()


def rotateRight(client):
    # Rotate UAV approx 30 deg to the right (clockwise)
    client.rotateByYawRateAsync(30,1).join()
    # Stop rotation
    client.rotateByYawRateAsync(0, 1e-6).join()


def moveForward(client):
    q = client.simGetGroundTruthKinematics().orientation
    yaw = quaternion2Yaw(q)
    # Calc velocity vector of magnitude 1 in direction of UAV
    vel = (np.cos(yaw), np.sin(yaw), 0) # Keep z fixed for now
    # Move forward approx 1 meter
    client.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=1).join()
    # Stop the UAV
    client.moveByVelocityAsync(0,0,0, duration=1e-6).join()

    # Debugging message
    print('Calculated yaw is {} deg'.format(yaw))


def printInfo(client):
    pos = client.simGetGroundTruthKinematics().position
    q = client.simGetGroundTruthKinematics().orientation
    yaw = quaternion2Yaw(q)

    print('Current yaw is {} deg'.format(yaw))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Start at origin
client.moveToPositionAsync(0,0,0,5).join()
printInfo(client)

#Make 20 random moves
for i in range(20):
    airsim.wait_key('Press any key to make next move')
    r = np.random.rand()
    if r < 0.33:
        print('Moving forward')
        moveForward(client)
    elif (r < 0.667):
        print('Rotating left')
        rotateLeft(client)
    else:
        print('Rotating right')
        rotateRight(client)
	
    printInfo(client)



airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)