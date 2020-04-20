import numpy as np
from .utils import get_orientation
from .utils import hover


# Actions
def rotate_left(client, duration=0.5, rate=20):
    # Rotate UAV approx 10 deg to the left (counter-clockwise)
    client.rotateByYawRateAsync(-rate, duration).join()
    # Stop rotation
    client.rotateByYawRateAsync(0, 1e-6).join()


def rotate_right(client, duration=0.5, rate=20):
    # Rotate UAV approx 10 deg to the right (clockwise)
    client.rotateByYawRateAsync(rate, duration).join()
    # Stop rotation
    client.rotateByYawRateAsync(0, 1e-6).join()


def move_forward(client):
    # AirSim is buggy and looses alt if z-vel is 0
    keep_altitude_z_vel = -3e-3
    yaw = get_orientation(client)

    # Calc velocity vector of magnitude 0.5 in direction of UAV
    vel = (0.5*np.cos(yaw), 0.5*np.sin(yaw), 0)
    # Move forward approx 0.25 meter
    client.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=0.5).join()
    # Stop the UAV, small z-vel to keep altitude
    client.moveByVelocityAsync(0, 0, -3e-3, duration=1e-6).join()


def move_up(client, velocity=0.5, duration=0.5):
    # Move up approx 0.25 m. Note direction of z-axis.
    client.moveByVelocityAsync(0, 0, -velocity, duration=duration).join()
    # Stop the UAV
    client.moveByVelocityAsync(0, 0, -3e-3, duration=1e-6).join()


def move_down(client, velocity=0.5, duration=0.5):
    # Move down approx 0.25 m. Note direction of z-axis.
    client.moveByVelocityAsync(0, 0, velocity, duration=duration).join()
    # Stop the UAV
    client.moveByVelocityAsync(0, 0, 0, duration=1e-3).join()
