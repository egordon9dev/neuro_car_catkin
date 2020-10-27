FROM ros:noetic-ros-base

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl vim tmux python3-pip
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) \
main" > /etc/apt/sources.list.d/ros-latest.list' && apt-key adv --keyserver \
'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL http://get.gazebosim.org | sh
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-noetic-catkin ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control

WORKDIR /home/rg/neuro_car_catkin
COPY . /home/rg/neuro_car_catkin
RUN mkdir -p /root/.gazebo/models/urdf
COPY urdf /root/.gazebo/models/urdf

RUN pip3 install --no-cache-dir -r requirements.txt

RUN /bin/bash -c 'source /opt/ros/noetic/setup.sh && catkin_make'

EXPOSE 11345

CMD ["./run_neurocar"]
