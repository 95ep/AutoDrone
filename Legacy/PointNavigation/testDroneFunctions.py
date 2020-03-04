import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from agent import *


def printInfo(currentClient):
    pos = currentClient.simGetGroundTruthKinematics().position
    q = currentClient.simGetGroundTruthKinematics().orientation
    yawDeg = airsim.to_eularian_angles(q)[2]/np.pi*180

    print('Current yaw is {} deg'.format(yawDeg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Wait a little while to make sure info is correct
time.sleep(1)
printInfo(client)

# Prepare some nice plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)

#Make 40 random moves
for i in range(50):
    #airsim.wait_key('Press any key to make next move')
    r = np.random.rand()
    if r < 1.2:
        print('Moving forward')
        moveForward(client)
    elif (r < 0.4):
        print('Rotating left')
        rotateLeft(client)
    elif (r < 0.6):
        print('Rotating right')
        rotateRight(client)
    elif (r < 0.8):
        print('Move up')
        moveUp(client)
    else:
        print('Move down')
        moveDown(client)

    # Wait a little while to make sure info is correct
    time.sleep(0.1)
    printInfo(client)

    # Plot depth and camera
    rgb, depthPerspective = getImages(client)
    ax1.imshow(rgb)
    ax2.imshow(depthPerspective)
    plt.show()
    plt.pause(.001)


client.moveByVelocityAsync(0,0,0,0.1).join()
airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
