import airsim
import numpy as np
import time


def get_position(client):
    """
    Makes call to AirSim and returns position of UAV
    :param client: AirsimEnv object
    :return: position of UAV
    """
    return client.simGetGroundTruthKinematics().position


def get_orientation(client):
    """
    Makes call to AirSim and returns orientation (yaw angle)
    :param client: AirsimEnv object
    :return: Orientation (yaw) of agent in radians.
    """
    q = client.simGetGroundTruthKinematics().orientation
    return airsim.to_eularian_angles(q)[2]


def get_compass_reading(client, target_position):
    """
    Calculates distance and relative orientation to target position
    :param client: AirsimEnv object
    :param target_position: Position of target. np array of shape (2,)
    :return: array containing distance and angle
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
    """
    Makes call to AirSim and extract depth and/or RGB images from the UAV's camera. Due to a bug in AirSim
    an empty array is sometimes returned which is caught in this method.
    :param client: AirsimEnv object
    :param sensor_types: List of the image types to extract. 'rgb' and 'depth' available.
    :param max_dist: Max depth at which the depth map is capped.
    :param height: height of the images
    :param width: width of the images
    :return: dict containing the 'rgb' and/or 'depth' images
    """
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
    """
    Calculates 3D positions based on depth map and pixel coordinates. Points with depth larger
    than max_dist is not included.
    :param points_2d: Pixel coordinates of the points
    :param depth: Depth map
    :param max_dist: Max distance that is considered
    :param field_of_view: Field of view (radians) of the camera
    :return: Corresponding 3D coordinates, local coord. sys., of the 2d points
    """
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
    """
    Transforms the 3D points from the local to the global coordinate system.
    :param point_cloud: 3D points in local coord. sys.
    :param client: AirsimEnv object
    :return: 3D points in global coord. sys
    """
    # Get position and orientation of client
    pos_vec = get_position(client)
    client_pos = np.array([[pos_vec.x_val, pos_vec.y_val, pos_vec.z_val]])
    client_orientation = client.simGetGroundTruthKinematics().orientation
    pitch, roll, yaw = airsim.to_eularian_angles(client_orientation)

    # Account for camera offset
    offset = np.array([0.46, 0, 0])
    point_cloud = point_cloud + offset

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

def valid_spawn(scene):
    """
    Generates a valid spawn point. If the scene is not basic23 the valid spawn is same as valid target.
    :param scene: The AirSim scene
    :return: Numpy array with x and y coord. of the spawn point
    """
    if scene == 'basic23':
        r = np.random.rand()
        if r < 1/5:
            target_x = 4.2 - np.random.rand() * 1
            target_y = 4.7 - np.random.rand() * 9
        elif r < 2/5:
            target_x = 10.2 - np.random.rand() * 1
            target_y = 4.7 - np.random.rand() * 9
        elif r < 3/5:
            target_x = 16.2 - np.random.rand() * 1
            target_y = 4.7 - np.random.rand() * 9
        elif r < 4/5:
            target_x = 21.1
            target_y = 4.7 - np.random.rand() * 9
        else:
            target_x = 20.9 - np.random.rand() * 21.9
            target_y = 1 - np.random.rand() * 2.6

        return np.array([target_x, target_y])
    else:
        return valid_trgt(scene)


def valid_trgt(scene):
    """
    Generates a valid target position at random.
    :param scene: The AirSim scene
    :return: Numpy array with x and y coord. of the target point
    """
    r = np.random.rand()
    if scene == "subT":
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

    elif scene == "basic23":
        target_x = 21 - 22 * np.random.rand()
        target_y = 5 - 9 * np.random.rand()
        target = np.array([target_x, target_y])
    else:
        raise ValueError("Scene not recognized")

    return target


def invalid_trgt(scene):
    """
    Generates an invalid target position at random.
    :param scene: The AirSim scene
    :return: Numpy array with x and y coord. of the target point
    """
    r = np.random.rand()
    if scene == "subT":
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

    elif scene == "basic23":
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
    """
    Checks if the agent terminated at target position, or if the target was invalid (unreachable).
    :param client: AirsimEnv object
    :param target_position: coordinate of the target position
    :param valid_trgt: Boolean indicating if the target is reachable or not
    :param threshold: Threshold for considering the target to be reached.
    :return: Boolean indicating if agent terminated at target position
    """
    if valid_trgt:
        compass = get_compass_reading(client, target_position)
        distance_to_target = compass[0]
        success = distance_to_target < threshold
    else:
        success = True
    return success


def generate_target(client, max_target_distance, scene=None, invalid_prob=0.0):
    """
    Generates a new goal target for the agent to reach. If scene string is None random position relative to current
    position of the UAV. Else random valid or invalid from predefined areas.
    :param client: AirsimEnv object
    :param max_target_distance: max distance to random target in case scene string being None.
    :param scene: The AirSim scene
    :param invalid_prob: Probability of the generated target being invalid, given scene string is not None.
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
    """
    Called from the reset method in AirsimEnv. This method resets the AirSim simulation, respawns the UAV etc.
    :param client: AirsimEnv object
    :param scene: The AirSim scene
    :return: None
    """
    client.reset()
    if scene is not None:
        time.sleep(0.2)
        pose = client.simGetVehiclePose()
        start_pos = valid_spawn(scene)
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
    """
    Sends the UAV approx. 0.5 m up in the air.
    :param client: AirsimEnv object
    :param v_z: Velocity in z direction.
    :return: None
    """
    client.moveByVelocityAsync(0, 0, v_z, duration=0.5).join()


def hover(client):
    """
    Keeps the UAV stationary
    :param client: AirsimEnv object
    :return:
    """
    client.moveByVelocityAsync(0, 0, 0, duration=1e-6).join()


def print_info(client):
    """
    Prints some info about the kinematics of the UAV.
    :param client: AirsimEnv object
    :return:
    """
    pos = get_position(client)
    orientation = get_orientation(client)
    yaw_deg = orientation/np.pi*180

    print('Current yaw is {} deg'.format(yaw_deg))
    print('Current position is ({}, {}, {})'.format(pos.x_val, pos.y_val, pos.z_val))
