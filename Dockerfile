FROM ros:noetic-ros-base

RUN apt-get update && apt-get install --no-install-recommends -y tmux python3-pip

WORKDIR .

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 11311
