import airsim
import numpy as np
import time
from Point2Point.agentActions import rotateLeft, rotateRight, moveForward
from Point2Point.util import printInfo


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Start at origin
client.moveToPositionAsync(0,0,0,5).join()
# Wait a little while to make sure info is correct
time.sleep(3)
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
	
    # Wait a little while to make sure info is correct
    time.sleep(2.5)
    printInfo(client)

airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)