import airsim
import numpy as np
import matplotlib.pyplot as plt


def getImages(currentClient, maxDist = 30):
    # Returns one rbg image and one depth mao
    responses = currentClient.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, pixels_as_float = False, compress = False),
        airsim.ImageRequest('front_center', airsim.ImageType.DepthPerspective, pixels_as_float = True, compress=False)])

    # Convert to uint and reshape to matrix with 3 color channels
    bgr = np.reshape(airsim.string_to_uint8_array(responses[0].image_data_uint8), (responses[0].height, responses[0].width, 3))
    # Move color channels around
    rgb = bgr[:,:,[2,1,0]]

    # Convert to 2D numpy array
    depthPerspective = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)

    return rgb, depthPerspective





# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
#client.takeoffAsync().join()

airsim.wait_key('Press any key capture images')

rgb, depthPerspective = getImages(client)

print(np.amax(rgb))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(rgb)
ax2.imshow(depthPerspective,vmax=100)

plt.show()
