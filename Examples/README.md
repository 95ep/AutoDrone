# Examples
This folder contains a number of examples illustrating the functionality of this repo.

## human_airsim_control
This script allows the user to manually control the UAV in the AirSim env. This script can be used to evaluate to total
return for a "human expert" as well as saving RGB and depth images to the computer. The images can be useful for 
evaluating computer vision algorithms.

The UAV is controlled with the keyboard and the following commands are available:
- a - rotate left
- d - rotate right
- w - move forward
- z - descend
- x - ascend
- s - terminate at waypoint
- t - exit script
- b - exit script
- c - save RGB image from UAV's camera
- f - save depth image from UAV's camera

The script is started with the following command. The image number argument is the number appended to the first captured
image, in order to avoid overwriting already saved images.

    python human_airsim_control.py --image_number 0 --parameters Parameters/parameters_airsim.json

## sift_demo
SIFT demo is a slightly modified version of the code in the object detection module. This demo does not calculate 3D
position but instead draws a bounding box of the object based on one or multiple reference images. This demo can be useful 
when evaluating if the SIFT algorithm is able to recognize some object. In this work we used the computer monitors from 
the Viktoria office.

Training and reference images can be obtained using the human_airsim_control script also include in the examples.
To run the SIFT demo script simply insert the proper paths to the reference and training images in the main section of the file.


