import airsim


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Start at origin
client.moveToPositionAsync(0,0,0,3)

# Rotate 30 deg left
client.rotateYawAsynd(-30).join()

# Move forward
client.moveByVelocityAsync(1, 0, 0, 3).join()

# Move up
client.moveByVelocityAsync(0, 0, 1, 3).join()

# Rotate 30 deg right
client.rotateYawAsynd(30).join()


# Disarm, reset and disconnect
client.armDisarm()
client.reset()
client.enableApiControl(False)