# neuro_car_catkin

# Docker

The docker container can be useful so you don't have to install all the dependencies on your local machine. It's based on the ROS Noetic base image: [noetic-ros-base](https://hub.docker.com/_/ros) It hasn't been tested with Gazebo yet though, so this may not work yet...

## build the docker container
`docker build -t rg/neurocar .`

## run the docker container
`docker run -it rg/neurocar /bin/bash`
