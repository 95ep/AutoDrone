import airsim
import numpy as np
import time
import random

# client.getMultirotorState() returns essentially everything. Maybe switch to that?


def get_position(client):
    return client.simGetGroundTruthKinematics().position


def get_orientation(client):
    """

    :param client:
    :return: Orientation of agent in radians.
    """
    q = client.simGetGroundTruthKinematics().orientation
    return airsim.to_eularian_angles(q)[2]


def get_compass_reading(client, target_position):
    # TODO: include height/z-dim?. Yes but later
    """

    :param client:
    :param target_position: Position of target. np array of shape (2,)
    :return:
    """
    pos = get_position(client)
    orientation = get_orientation(client)
    direction_vector = np.array([target_position[0] - pos.x_val, target_position[1] - pos.y_val])
    if direction_vector[0] == 0 and direction_vector[1] == 0:
        return np.array([0, 0])

    u = np.array([np.cos(orientation), np.sin(orientation)])    # orientation vector
    v = direction_vector / np.linalg.norm(direction_vector)     # normalized target direction
    angle_mag = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    angle_sign = np.sign(np.cross(u, v))
    angle_sign = 1 if angle_sign == 0 else angle_sign
    dist = np.linalg.norm(direction_vector)
    return np.array([dist, angle_mag * angle_sign])


# Observations
def get_camera_observation(client, sensor_types=['rgb', 'depth'], max_dist=10, height=64, width=64):
    requests = []
    sensor_idx = {}
    idx_counter = 0
    if 'rgb' in sensor_types:
        requests.append(airsim.ImageRequest(
            'front_center', airsim.ImageType.Scene, pixels_as_float=False, compress=False))
        sensor_idx.update({'rgb': idx_counter})
        idx_counter += 1
    if 'depth' in sensor_types:
        requests.append(airsim.ImageRequest(
            'front_center', airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False))
        sensor_idx.update({'depth': idx_counter})
        idx_counter += 1

    responses = client.simGetImages(requests)

    images = {}
    if 'rgb' in sensor_types:
        idx = sensor_idx['rgb']
        # convert to uint and reshape to matrix with 3 color channels
        try:
            bgr = np.reshape(airsim.string_to_uint8_array(
                responses[idx].image_data_uint8), (height, width, 3))
            # move color channels around
            rgb = np.array(bgr[:, :, [2, 1, 0]], dtype=np.float32)
        except ValueError as err:
            print('========================================================')
            print('Value err when moving color channels: {0}'.format(err))
            print('Replacing rgb with all zeros')
            print('========================================================')
            rgb = np.zeros((height, width), dtype=np.float32)
        images.update({'rgb': rgb})

    if 'depth' in sensor_types:
        idx = sensor_idx['depth']
        # convert to 2D numpy array. Had unexpected exception here. Try: Catch
        try:
            depth = airsim.list_to_2d_float_array(
                responses[idx].image_data_float, width, height)
        except ValueError as err:
            print('========================================================')
            print('Value err when converting depth image: {0}'.format(err))
            print('Replacing depth map with all max dist values')
            print('========================================================')
            depth = np.ones((height, width), dtype=np.float32)*max_dist

        depth = np.expand_dims(depth, axis=2)
        images.update({'depth': depth})

    return images


def has_collided(client, floor_z=0.5, ceiling_z=-4.5):

    collision_info = client.simGetCollisionInfo()
    if collision_info:
        client.simPrintLogMessage("Collision with object")
    z_pos = get_position(client).z_val
    if z_pos > floor_z:
        client.simPrintLogMessage("Collision with floor")
    if z_pos < ceiling_z:
        client.simPrintLogMessage("Collision with ceiling")

    return collision_info.has_collided or z_pos > floor_z or z_pos < ceiling_z


def target_found(client, target_position, threshold=0.5):
    compass = get_compass_reading(client, target_position)
    distance_to_target = compass[0]
    success = distance_to_target < threshold
    return success


def generate_target(client, max_target_distance, sub_t=False):
    """
    Generate new goal for the agent to reach.
    :param client:
    :param max_target_distance:
    :return:
    """
    if sub_t:
        targets = [[-15, -17],[-15, -28],[-3,-28,],[-15,-60],[-25,-61],[-36,-61],[-48,-63],[-15,-20],[-15,-50],[-15,-42],[-15,-30],[-48,-70],[-23,-50]]
        target = np.array(random.choice(targets), dtype=np.float)
    else:
        pos = client.simGetGroundTruthKinematics().position
        x = (2 * np.random.rand() - 1) * max_target_distance + pos.x_val
        y = (2 * np.random.rand() - 1) * max_target_distance + pos.y_val
        target = np.array([x, y])
    return target


def reset(client):
    client.reset()
    time.sleep(0.2)
    client.enableApiControl(True)
    client.armDisarm(True)
    hover(client)
    custom_takeoff(client)
    hover(client)


def custom_takeoff(client, z=-2.0):
    client.moveByVelocityZAsync(0, 0, z, duration=2).join()


def hover(client):
    client.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()
    client.takeoffAsync().join()


def print_info(client):
    pos = get_position(client)
    orientation = get_orientation(client)
    yaw_deg = orientation/np.pi*180

    print('Current yaw is {} deg'.format(yaw_deg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))