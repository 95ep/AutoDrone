# use open cv to create point cloud from depth image.
#import setup_path
import airsim

import cv2
import time
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


width = 64
height = 64
focal_len = width / (2*np.tan(np.pi/4))
centerX = width/2
centerY = height/2
max_dist = 40



client = airsim.MultirotorClient()

airsim.wait_key("Press any key to capture")
responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False)])
depth = airsim.list_to_2d_float_array(
    responses[0].image_data_float, responses[0].width, responses[0].height)

print("Shpe of depth {}".format(depth.shape))
print("Min and max {} {}".format(depth.min(), depth.max()))

h_idx = np.linspace(0, height, 8, endpoint=False, dtype=np.int)
w_idx = h_idx

H_IDX, W_IDX = np.meshgrid(h_idx, w_idx)

points = []
for v in h_idx:
    for u in w_idx:
        X = depth[v,u]
        if X < max_dist:
            Y = (u - centerX) * X / focal_len
            Z = (v - centerY) * X / focal_len
            points.append([X, Y, Z])

cleanPoints = np.array(points)
print("Points with obstacles:")
print(cleanPoints)
pos_vec = client.simGetGroundTruthKinematics().position
client_pos = np.array([[pos_vec.x_val, pos_vec.y_val, pos_vec.z_val]])
print("Client position is {}".format(client_pos))
client_orientation = client.simGetGroundTruthKinematics().orientation

print("Client orientation is {}".format(airsim.to_eularian_angles(client_orientation)))
# Angles to compensate by
pitch, roll, yaw = airsim.to_eularian_angles(client_orientation)
pitch, roll, yaw = -pitch, -roll, -yaw

rot_mat = np.array([
                    [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
                    ])

print("Compensated points")
global_points = rot_mat @ cleanPoints.transpose() + client_pos.transpose()
global_points = global_points.transpose()
print(global_points)


fig_depth = plt.figure()
ax1 = fig_depth.subplots()
ax1.imshow(depth, vmax = max_dist, cmap=plt.cm.gray)
ax1.plot(H_IDX, W_IDX, 'ro', markersize = 1)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(cleanPoints[:,0], cleanPoints[:,1], cleanPoints[:,2])
ax.view_init(elev=-170, azim=0)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_zlabel('z')
ax.set_xlim(0, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-15, 10)
plt.show()
