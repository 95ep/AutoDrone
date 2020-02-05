import airsim
import numpy as np


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
    # TODO: include height/z-dim?
    """

    :param client:
    :param target: Position of target. np array of shape (2,)
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
    return np.array([np.linalg.norm(direction_vector), angle_mag * angle_sign])


# Observations
def get_camera_observation(client, sensor_types=['rgb', 'depth'], max_dist=10):
    requests = []
    sensor_idx = {}
    idx_counter = 0
    if 'rbg' in sensor_types:
        requests.append(airsim.ImageRequest(
            'front_center', airsim.ImageType.Scene, pixels_as_float=False, compress=False))
        sensor_idx.update({'rbg': idx_counter})
        idx_counter += 1
    if 'depth' in sensor_types:
        requests.append(airsim.ImageRequest(
            'front_center', airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
        sensor_idx.update({'depth': idx_counter})
        idx_counter += 1

    responses = client.simGetImages(requests)

    images = {}

    if 'rbg' in sensor_types:
        idx = sensor_idx['rgb']
        # convert to uint and reshape to matrix with 3 color channels
        bgr = np.reshape(airsim.string_to_uint8_array(
            responses[idx].image_data_uint8), (responses[idx].height, responses[idx].width, 3))
        # move color channels around and change representation to float in range [0, 1]
        rgb = np.array(bgr[:, :, [2, 1, 0]], dtype=np.float32)/255
        images.update({'rgb': rgb})

    if 'depth' in sensor_types:
        idx = sensor_idx['depth']
        # convert to 2D numpy array
        depth = airsim.list_to_2d_float_array(
            responses[idx].image_data_float, responses[idx].width, responses[idx].height)
        # clip array after max_dist
        depth = np.clip(depth, None, max_dist)
        images.update({'depth': depth})

    return images


def print_info(client):
    pos = get_position(client)
    orientation = get_orientation(client)
    yaw_deg = orientation/np.pi*180

    print('Current yaw is {} deg'.format(yaw_deg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))


def generate_target(client):
    """
    Generate new goal for the agent to reach.
    :param client:
    :return:
    """
    pos = client.simGetGroundTruthKinematics().position
    x = np.random.rand()*40 -20.0 + pos.x_val
    y = np.random.rand()*40 -20.0 + pos.y_val
    return np.array([x, y])