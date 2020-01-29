import airsim
import numpy as np
import time
from agentActions import rotateLeft, rotateRight, moveForward
from util import printInfo


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

#Make 20 random moves
for i in range(20):
    #airsim.wait_key('Press any key to make next move')
    r = np.random.rand()
    if r < 0.5:
        print('Moving forward')
        moveForward(client)
    elif (r < 0.75):
        print('Rotating left')
        rotateLeft(client)
    else:
        print('Rotating right')
        rotateRight(client)
	
    # Wait a little while to make sure info is correct
    time.sleep(0.1)
    printInfo(client)

client.moveByVelocityAsync(0,0,0,0.1).join()
airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)