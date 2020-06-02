# Map environment 

*map_env* contains a class *MapEnv* that is a combined dynamic map utility and reinforcement learning environment designed to keep track of an agent exploring a scene. 
It also contains an extension of the class called *AirSimMapEnv*, which combines the reinforcement learning environment 
with Airsim, making it possible to control a simulated UAV using high level commands in the map.

## Map Tool

*map_env* can be used to manage a cell/voxel map. Initiate a cell map using 
    
    from Environments.Exploration import map_env
    parameters = {}
    m = map_env.MapEnv(**parameters)
    m.reset()
   
   
where the *parameters* dictionary is filled out with relevant fields (explained below).
The map can assign different categories to the cells. From the start all cells are *unknown*, except for the center cell which 
is the current *position*. The map can simulate the field of view of the agent (controlled by the *fov_angle* and *vision_range* parameters).
Whenever the map is updated with the new position of the agent and its orientation by calling the function
    
    m._update(new_position, orientation=None, detected_obstacles=(), detected_objects=())

every cell that falls within this field of view is marked *visible* in the map. If no orientation is given, the orientation
is calculated based on the movement direction from the last step. In addition, it is possible to report detections of obstacles 
and target objects, which will mark the corresponding cells in the map with the labels *obstacle* and *object*. Whenever 
the agent moves, a new cell will represent the *position* and the previous cell with this label is changed to *visited*.
Which categories are used can be specified by the *map_keys* parameter.

The map can be visualized using the *render* function

    m.render(render_3d=False, local=False, num_ticks_approx=6, show_detected=False, voxels=True, ceiling_z=None, floor_z=None)

with keyword arguments
* *render_3d* - bool: if True show a 3D visualization, else render in 2D.
* *local* - bool: If True, crop the map to only include the immediate surroundings of the agent, as specified in the parameters.
* *show_detected* - bool: show cells that are *visited* if True, else omit them and consider them *unknown*.
* *voxels* - bool: whether to plot using cubic voxels, else plot 3D points
* *ceiling_z* - float: crop the map at specified height, can be used to remove ceiling detections.
* *floor_z* - float: crop the map at specified height, can be used to remove floor detections.

### Additional functionality
The map will increase in size automatically when necessary. The size increase can be triggered by the agent moving too close
to a border in some dimension, or when trying to access a cell that is currently outside the map. The first scenario is controlled
by the *buffer_distance* parameter (see below). Mostly this is useful if there is no need to increase the map in some dimension,
e.g. if the height of the environment is known, it can be specified in the *starting_map_size*. Then the z-value of the *buffer_distance* 
can be set to 0 to keep the map from growing in that direction.

A local crop of the map can be obtained. This can be useful e.g. if other systems need to specify the size
of an input because the full map may grow dynamically. The size of the local map is specified by *local_map_dim*.

Thresholds for different detections can be set in order to force several detections in the same cell before the map assigns the 
corresponding label. E.g. if 

    thresholds={'visible': 3, 'obstacle': 5}
   
then a cell must be visible to the agent for a total of 3 different time steps before the cell is marked as *visible*.

## Reinforcement Learning Environment

A reinforcement learning environment following the Gym API can be accessed on top of the map. This environment can be used
to solve an exploration task. There are several predefined maze-like maps (chosen by the *map_idx* parameter) that are created by *training_maps.py*. 
The task is then to produce waypoints in the environment that a naive controller (which moves the agent straight towards a waypoint) 
can reach. When the agent moves, its field of view is used to mark 'unknown' cells as 'visible' along its way. The agent
is rewarded depending on how many cells are detected on the way to the waypoint.

After creating and resetting the environment (as described above) the map, the step function can be called to update the map.

    m.step(action, safe_mode=False)

The next waypoint is given as the *action*. *safe_mode* controls whether the naive controller should be able to collide with
obstacles or not. If activated (True), bad waypoints can be mitigated by stopping before a collision occurs. Calling *step*
returns a tuple: (observation, reward, done, info), where observation is a local map of the surroundings, the reward depends 
on the number of detected cells, done is a bool which is True if the agent collided and the episode is over, info is a dictionary
with additional information. 

## Parameters

* *starting_map_size*: map size in meters, array like e.g. (10., 10., 2.)
* *cell_scale*: length of each cell in meters, array like e.g. (1., 1., 0.5)
* *starting_position*: starting position of the agent, array like e.g. (0., 0., 1.)
* *buffer_distance*: the furthest distance from a border that triggers extension of the map.
* *local_map_dim*: number of cells in the local map. Manually enter starting_map_size and cell_scale to match wanted output shape.
* *vision_range*: vision range in meters. Recommended that it is smaller than buffer_distance, but not necessary. If no vision model is wanted, set = 0.
* *fov_angle*: field of view angle of the vision of the agent, in radians.
* *map_keys*: list of labels for which things to be tracked in the map and visualized: ['unknown', 'position', 'visited', 'visible', 'obstacle', 'object']
* *thresholds*: dictionary {'map_key': threshold_value}. Number of detections in a cell needed to mark the occurrence
* *map_idx*: (For RL environment) int in range [-1,9] to choose from predefined training maps, -1 to randomize index. 0 default empty map, suitable for Airsim.
* *interactive_plot*: Set to True only if rendering 2D pyplots and want the plot to update in the same window.
* *REWARD_FAILURE*: Reinforcement learning penalty for collisions
* *REWARD_STEP*: Reinforcement learning cost for defining a waypoint

## AirSimMapEnv

Combining the map with the Airsim drone simulation provides a richer environment. Instead of using a naive controller 
which is responsible for moving the agent in the map, *AirSimMapEnv* will use a (pre-trained) neural network agent which
steers the UAV in the simulated environment. In addition to the parameters described above, the class takes a dictionary
of keyword arguments as described in the airgym module. See provided parameter files for examples. When calling *step(action=(dx, dy))*
the local navigation module will move the drone in the simulated environment and report back to the map, adding different detections.



