# neuro_car_catkin 

# Docker

The docker container can be useful so you don't have to install all the dependencies on your local machine. It's based on the ROS Noetic base image: [noetic-ros-base](https://hub.docker.com/_/ros) It hasn't been tested with Gazebo yet though, so this may not work yet...

## build the docker container
`docker build -t rg/neurocar .`

## run the simulation in docker container (gzserver + ROS)
`docker run --rm -it rg/neurocar`

## view simulation live (gzclient)
in a seperate terminal:
`./run_gui`
* this requires Ubuntu 20.04 and ROS Noetic to be installed locally. you may also need to install these packages: `sudo apt install ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control ros-noetic-gazebo-ros-pkgs` 
