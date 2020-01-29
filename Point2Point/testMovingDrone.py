import airsim
import numpy as np
import time


def quaternion2Yaw(q):
    # Convert from quaternion to Yaw (according to Wikipedia)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp) /(2*np.pi)*360
    return yaw


def rotateLeft():
    q = client.simGetGroundTruthKinematics().orientation
    yaw = quaternion2Yaw(q)
    print('New yaw: ' + str(yaw - 30))
    client.rotateToYawAsync(yaw - 30,margin=0.1).join()
    print('Rotated left')


def rotateRight():
    q = client.simGetGroundTruthKinematics().orientation
    yaw = quaternion2Yaw(q)

    client.rotateToYawAsync(yaw + 30).join()
    print('Rotated right')


def moveForward(distance=1, vel=2):
    currentPos = client.simGetGroundTruthKinematics().position
    q = client.simGetGroundTruthKinematics().orientation
    yaw = quaternion2Yaw(q)
    newPos = (currentPos.x_val + distance * np.cos(yaw), currentPos.y_val + distance * np.sin(yaw), currentPos.z_val) # Keep z fixed for now

    client.moveToPositionAsync(newPos[0], newPos[1], newPos[2], vel, adaptive_lookahead=0).join()
    print('New pos')
    print(newPos)


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Start at origin
client.moveToPositionAsync(0,0,0,5).join()

# This seems to enable rotateToYaw()
client.rotateByYawRateAsync(20.0,1).join()
client.rotateToYawAsync(0.0,margin=1).join()

initPos = client.simGetGroundTruthKinematics().position
print('Initial position')
print(initPos)

airsim.wait_key('Press any key to start demo')

moveForward(distance=5, vel=2)
airsim.wait_key('Press key')
rotateLeft()
airsim.wait_key('Press key')
moveForward(distance=5, vel=2)
airsim.wait_key('Press key')
rotateLeft()

# Make 20 random moves
#for i in range(20):
#    time.sleep(1)
##    r = np.random.rand()
 #   if r < 0.33:
 #       print('Moving forward')
#        moveForward()
#    elif (r < 0.667):
#        print('Rotating left')
#        rotateLeft()
#    else:
#        print('Rotating right')
#        rotateRight()



airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)