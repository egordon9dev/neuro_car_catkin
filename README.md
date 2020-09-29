# neuro_car_catkin

clone this to `~/catkin_ws/src`<br>
to view logs: `rostopic echo neurocar/log`

### Docker ###

The docker container can be useful so you don't have to install all the dependencies on your local machine. It hasn't been tested with Gazebo yet though, so this may not work yet...

# build the docker container
`docker build -t rg/neurocar .`

# run the docker container
`docker run -it rg/neurocar /bin/bash`
