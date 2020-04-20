import airsim
import numpy as np
import time

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
            print('Value err when reshaping RGB image: {0}'.format(err))
            print('Replacing rgb with all zeros')
            print('========================================================')
            rgb = np.zeros((height, width, 3), dtype=np.float32)
        images.update({'rgb': rgb})

    if 'depth' in sensor_types:
        idx = sensor_idx['depth']
        # convert to 2D numpy array. Had unexpected exception here. Try: Catch
        try:
            depth = airsim.list_to_2d_float_array(
                responses[idx].image_data_float, width, height)
        except ValueError as err:
            print('========================================================')
            print('Value err when reshaping depth image: {0}'.format(err))
            print('Replacing depth map with all max dist values')
            print('========================================================')
            depth = np.ones((height, width), dtype=np.float32) * max_dist

        depth = np.expand_dims(depth, axis=2)
        images.update({'depth': depth})

    return images


def reproject_2d_points(points_2d, depth, max_dist, field_of_view):
    h, w = depth.shape
    center_x = w // 2
    center_y = h // 2
    focal_len = w / (2 * np.tan(field_of_view / 2))
    points = []
    for u, v in points_2d:
        x = depth[v, u]
        if x < max_dist:
            y = (u - center_x) * x / focal_len
            z = (v - center_y) * x / focal_len
            points.append([x, y, z])

    return np.array(points)


def local2global(point_cloud, client):
    # Get position and orientation of client
    pos_vec = get_position(client)
    client_pos = np.array([[pos_vec.x_val, pos_vec.y_val, pos_vec.z_val]])
    client_orientation = client.simGetGroundTruthKinematics().orientation
    pitch, roll, yaw = airsim.to_eularian_angles(client_orientation)
    # pitch, roll, yaw = -pitch, -roll, -yaw

    rot_mat = np.array([
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll),
         np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
        [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll),
         np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
                        ])

    # get global_points, i.e. compensate for client orientation and position
    global_points = rot_mat @ point_cloud.transpose() + client_pos.transpose()
    global_points = global_points.transpose()
    return global_points


def valid_trgt(env):
    r = np.random.rand()
    if env == "subT":
        if r < 0.2:
            target_x = 1 - np.random.rand()*31
            target_y = -119
        elif r < 0.4:
            target_x = 1 - np.random.rand()*31
            target_y = -127.5
        elif r < 0.6:
            target_x = -30
            target_y = -123.5 - np.random.rand()*6
        elif r < 0.8:
            target_x = -16
            target_y = -119 - np.random.rand()*18
        else:
            target_x = -1
            target_y = -119 - np.random.rand()*12
        target = np.array([target_x, target_y])

    elif env == "basic23":
        if r < 1/6:
            target_x = 20.5 - 21.5 * np.random.rand()
            target_y = 0
        else:
            target_y = 5 - 9 * np.random.rand()
            if r < 2/5:
                target_x = 3.8
            elif r < 3/5:
                target_x = 9.4
            elif r < 4/5:
                target_x = 15.5
            else:
                target_x = 20.5
        target = np.array([target_x, target_y])
    else:
        raise ValueError("Env not recognized")

    return target


def invalid_trgt(env):
    r = np.random.rand()
    if env == "subT":
        if r < 0.2:
            target_x = 1 - np.random.rand()*31
            target_y = -117 + np.random.rand()*3
        elif r < 0.4:
            target_x = -4 - np.random.rand()*4
            target_y = -125.5 + np.random.rand()*3
        elif r < 0.6:
            target_x = -22 - np.random.rand()*4
            target_y = -125.5 + np.random.rand()*3
        elif r < 0.8:
            target_x = -4 - np.random.rand()*4
            target_y = -131 - np.random.rand()*5
        else:
            target_x = -22 - np.random.rand()*4
            target_y = -131 - np.random.rand()*5
        target = np.array([target_x, target_y])

    elif env == "basic23":
        if r < 1/4: # South sector
            target_x = -2 - 15 * np.random.rand()
            target_y = 16 - 32 * np.random.rand()
        elif r < 2/4: # North sector
            target_x = 22 + 15 * np.random.rand()
            target_y = 16 - 32 * np.random.rand()
        elif r < 3/4: # West sector
            target_x = 22 - 24 * np.random.rand()
            target_y = -6 - 15 * np.random.rand()
        else: # East sector
            target_x = 22 - 24 * np.random.rand()
            target_y = 7 + 15 * np.random.rand()
        target = np.array([target_x, target_y])
    else:
        raise ValueError("Env not recognized")

    return target


def target_found(client, target_position, valid_trgt = True, threshold=0.5):
    if valid_trgt:
        compass = get_compass_reading(client, target_position)
        distance_to_target = compass[0]
        success = distance_to_target < threshold
    else:
        success = True
    return success


def generate_target(client, max_target_distance, scene=None, invalid_prob=0.0):
    """
    Generate new goal for the agent to reach.
    :param client:
    :param max_target_distance:
    :return:
    """
    is_valid = True
    if scene is not None:
        r = np.random.rand()
        if r < invalid_prob:
            target = invalid_trgt(scene)
            is_valid = False
        else:
            target = valid_trgt(scene)

    else:
        pos = client.simGetGroundTruthKinematics().position
        x = (2 * np.random.rand() - 1) * max_target_distance + pos.x_val
        y = (2 * np.random.rand() - 1) * max_target_distance + pos.y_val
        target = np.array([x, y])
    return target, is_valid


def reset(client, scene=None):
    client.reset()
    if scene is not None:
        time.sleep(0.2)
        pose = client.simGetVehiclePose()
        start_pos = valid_trgt(scene)
        pose.position.x_val = start_pos[0]
        pose.position.y_val = start_pos[1]
        pose.position.z_val = 0
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        yaw = np.random.rand() * 2 * np.pi
        pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
        client.simSetVehiclePose(pose, True)

    time.sleep(0.2)
    client.enableApiControl(True)
    client.armDisarm(True)
    hover(client)
    custom_takeoff(client)
    hover(client)


def custom_takeoff(client, v_z=-1.0):
    client.moveByVelocityAsync(0, 0, v_z, duration=0.5).join()


def hover(client):
    client.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()


def print_info(client):
    pos = get_position(client)
    orientation = get_orientation(client)
    yaw_deg = orientation/np.pi*180

    print('Current yaw is {} deg'.format(yaw_deg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))
