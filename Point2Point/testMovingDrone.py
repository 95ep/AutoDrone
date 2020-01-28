import airsim


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Start at origin
client.moveToPositionAsync(0,0,0,10).join()


airsim.wait_key('Press any key to move forward')
selfPos = client.simGetGroundTruthKinematics().position



airsim.wait_key('Press any key to go down')
client.moveToPositionAsync(0,0,10,3).join()

# Move forward
airsim.wait_key('Press any key to move forward')
client.moveByVelocityAsync(1, 0, 0, 3).join()
print('Hovering')
client.hoverAsync().join()

airsim.wait_key('Press any key test rotation')

print('Rotate to 30 deg using YawMode')
client.moveByVelocityAsync(0, 0, 0, 5, yaw_mode=airsim.YawMode(False, 30)).join()


print('Rotate to 120 deg')
client.moveByVelocityAsync(0, 0, 0, 5, yaw_mode=airsim.YawMode(False, 120)).join()


print('Rotate to 0 deg')
client.moveByVelocityAsync(0, 0, 0, 5, yaw_mode=airsim.YawMode(False, 0)).join()

print('Rotate to 160 deg')
client.moveByVelocityAsync(0, 0, 0, 0, yaw_mode=airsim.YawMode(False, 160)).join()

# Move up
airsim.wait_key('Press any key to move up')
client.moveByVelocityAsync(0, 0, -1, 2).join()
client.hoverAsync().join()


airsim.wait_key('Press any key to land and disconnect')
# Disarm, reset and disconnect
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)