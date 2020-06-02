# Airgym #
Airgym containes the code for wrapping AirSim in order to make it suitable for Reinforcement Learning. 
The AirsimEnv class is the main attraction and extends the Env class of OpenAI Gym.
Addition files contains helper functions to the AirsimEnv class.

The content of the different files are summarized briefly below and more information can be found in 
the docstrings in the files.

## AirsimEnv
*airsim_env.py* contains the definition of AirSimEnv and a number of object methods. 
Most methods facilitates the use of this environment for RL tasks, but methods for 
obstacle and object detection is also part of this class.

## agent_controller
The methods defined in this file communicates with AirSim and carries out the movement actions. 
Due to some unexpected behavior from AirSim, probably due to the built in PID-controller, some 
unorthodox methods had to be used to ensure that the UAV maintained it's position.

## utils
This file contains a number of utility functions used by the AirsimEnv class. Most are functions for exchanging 
information with AirSim, such as extracting sensor data or spawning the UAV at custom positions. The functions 
for generating target positions are also placed in this script.

Finally, this file also contains functions for reprojecting a pixel in an image from the camera to its corresponding
3D point. These functions are used by the obstacle and object detection algorithms.

## Example Use
Below follows some examples on how central parts of airgym is used. In order to create this env an instance of AirSim
must be running in Unreal.


    import airgym
    
    # Create env and take a step
    env = airgym.make(sensors=['rgb', 'depth', 'pointgoal_with_gps_compass'], max_dist=10)
    obs = env.reset()
    observation, reward, episode_over, info = env.step(action=1)
   
    # Get obstacles from depth camera
    obstacles_coords = env.get_obstacles(field_of_view, n_gridpoints=8):
   
    # Init object detection
    query_paths = ['put_one_or_multiple_paths_to_reference_image_here']
    setup_object_detection(self, query_paths, rejection_factor=0.8, min_match_thres=10)
    
    Match reference image(s) with image from the camera on the UAV
    objects_coords = env.get_trgt_objects()
```