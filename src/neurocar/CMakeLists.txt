cmake_minimum_required(VERSION 2.8.3)
project(neurocar)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  gazebo_msgs
  roscpp
  rospy
  std_msgs
)
catkin_python_setup()
catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/train.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)