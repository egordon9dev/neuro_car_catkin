FROM ros:noetic-ros-base

RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-noetic-catkin \
    curl vim tmux python3-pip
RUN curl -sSL http://get.gazebosim.org | sh

WORKDIR .
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN /bin/bash -c 'source /opt/ros/noetic/setup.sh && catkin_make'

EXPOSE 11311
EXPOSE 11345

CMD ["./run_neurocar"]
