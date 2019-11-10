#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np
import math
import nengo
import nengo_fpga
from nengo_fpga.networks import FpgaPesEnsembleNetwork

bridge = CvBridge()


min_range = 10000
min_range_angle = 0
max_range = 0
max_range_angle = 0
lasers = 100 * np.ones(6)
prev_lasers = 100 * np.ones(6)


def laser_callback(laser_scan):
    global min_range, min_range_angle, max_range, max_range_angle, lasers
    avg_range = 0
    min_range = 100000
    min_range_angle = 0
    max_range = 0
    max_range_angle = 0
    prev_lasers = lasers
    if len(laser_scan.ranges) > 0:
        lasers = laser_scan.ranges
        for i, rng in enumerate(laser_scan.ranges):
            avg_range += rng
            current_angle = laser_scan.angle_min + i * laser_scan.angle_increment
            if rng < min_range:
                min_range = rng
                min_range_angle = current_angle
            if rng > max_range:
                max_range = rng
                max_range_angle = current_angle
        avg_range /= len(laser_scan.ranges)
    # log_string.data = "closest obstacle --- min " + str(min_range_angle) + " " + str(min_range) + " max " + str(max_range_angle) + " " + str(max_range)


width = 1280
height = 720

shrink_width = 64
shrink_height = 36

angular_vel = 0
trans_vel = 0
img_arr = np.ones(shrink_width*shrink_height)
log_msg = String()
real_twist = Twist()
# acceleration: translational, rotational
actions_list = [
    # [[-1], [-1]],
    [[-1], [0]],
    # [[-1], [1]],
    [[0], [-1]],
    # [[0], [0]],
    [[0], [1]],
    # [[1], [-1]],
    [[1], [0]],
    # [[1], [1]]
]


def callback(img):
    global img_arr, width, height
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    cv_image = cv2.resize(cv_image, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)
    # edges_img = cv2.Canny(cv_image, 50, 250)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([255,255,10])
    thresh = cv2.inRange(hsv, lower, upper)

    img_arr = np.asarray(thresh).flatten().clip(0,1).astype(int)

    blown_up = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image window", blown_up)
    cv2.waitKey(3)

def odom_callback(odom):
    global real_twist
    real_twist = odom.twist.twist

sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
ray_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laser_callback)
odom_sub = rospy.Subscriber('neurocar/odom', Odometry, odom_callback)
pub_log = rospy.Publisher('neurocar/log', String, queue_size=10)
pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
rospy.init_node('controller', anonymous=True)
rate = rospy.Rate(60)  # 10hz
neurocar_msg = Twist()

def move(t, x):
    global angular_vel, trans_vel, min_range
    global lasers, prev_lasers

    prev_min_range = min_range

    trans_vel += x[0]
    angular_vel += x[1]
    trans_vel = max(min(trans_vel, 2), -2)
    angular_vel = max(min(angular_vel, 1), -1)
    if min(lasers) < 0.5 or math.isinf(max(lasers)):
        trans_vel = 0


    # send action
    neurocar_msg.linear.x = trans_vel
    neurocar_msg.angular.z = angular_vel
    pub.publish(neurocar_msg)
    pub_log.publish(log_msg)

    # do action
    rate.sleep()

    reward = 0
    # move away from obstacles

    # delta_lasers = np.absolute(np.subtract(lasers, prev_lasers))
    max_laser = max(lasers)
    if math.isfinite(max_laser):
        reward += (max_laser-10) * 10

    if min(lasers) > 1:
        reward += 10
        if min(prev_lasers) < 1:
            reward += 100

    reward += abs(real_twist.linear.x) * 1000

    log_msg.data = "reward: " + str(reward)
    return reward


class InputManager:
    """
    The InputManager maintains the current input data for a neural network.
    To update the data, please call fn and ensure that
    the data dimensionality is exactly the same as the previous data.
    """

    def __init__(self):
        global shrink_width, shrink_height, lasers
        self.dimensions = len(img_arr) #len(lasers)

    def function(self, t):
        global img_arr
        return img_arr

class NeuralNet:
    """
    A NeuralNet contains a Nengo network with an FPGA implementation.
    It intends to receive data through an InputManager.
    It will select one of a few actions and execute a given function.
    """

    def __init__(self, input_manager, act_function, learning_active=1, board="pynq", learn_rate=1e-4,
                 learn_synapse=0.030, action_threshold=0.1, init_transform=[0, 0, 0, 1]):
        global actions_list
        self.model = nengo.Network()
        self.input_manager = input_manager
        self.learning_active = learning_active
        self.board = board
        # parameters for the learning model (i.e. i don't know what they really do)
        self.learn_rate = learn_rate
        self.learn_synapse = learn_synapse
        self.action_threshold = action_threshold
        self.init_transform = init_transform

        with self.model:
            # Set up the movement node
            movement = nengo.Ensemble(n_neurons=100, dimensions=2)
            movement_node = nengo.Node(act_function, size_in=2, label="reward")
            nengo.Connection(movement, movement_node)

            # set up input stimulus
            stim_node = nengo.Node(self.input_manager.function)
            stim_ensemble = nengo.Ensemble(n_neurons=300, dimensions=self.input_manager.dimensions)
            nengo.Connection(stim_node, stim_ensemble)

            # Create the action selection networks
            basal_ganglia = nengo.networks.actionselection.BasalGanglia(len(actions_list))
            thalamus = nengo.networks.actionselection.Thalamus(len(actions_list))
            nengo.Connection(basal_ganglia.output, thalamus.input)

            # Convert the selection actions to act transforms
            conn_actions = []
            for i, action in enumerate(actions_list):
                conn_actions.append(nengo.Connection(stim_ensemble, basal_ganglia.input[i], function=lambda x: 0, learning_rule_type=nengo.PES()))
                nengo.Connection(thalamus.output[i], movement, transform=action)

            errors = nengo.networks.EnsembleArray(n_neurons=100, n_ensembles=len(actions_list))
            nengo.Connection(movement_node, errors.input, transform=-np.ones((len(actions_list),1)))
            for i in range(len(actions_list)):
                nengo.Connection(basal_ganglia.output[i], errors.ensembles[i].neurons, transform=np.ones((100, 1)) * 4)
            nengo.Connection(basal_ganglia.input, errors.input, transform=1)
            for i in range(len(actions_list)):
                nengo.Connection(errors.ensembles[i], conn_actions[i].learning_rule)

        self.simulator = nengo_fpga.Simulator(self.model)

    def run_network(self, number_of_seconds):
        # with self.simulator:
        self.simulator.run(number_of_seconds)

    def step_network(self):
        # with self.simulator:
        self.simulator.step()

    def close_simulator(self):
        self.simulator.close()


# print("Syntax correct.")
def main():
    input_manager = InputManager()
    network = NeuralNet(input_manager, move)
    network.run_network(60)
    network.close_simulator()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



"""
To run:

cd ~/catkin_ws/src
catkin_make -C ~/catkin_ws
source ~/catkin_ws/devel/setup.bash
source /usr/share/gazebo/setup.sh
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws/src
export ROS_PYTHON_VERSION=3

cd ~/catkin_ws/src
source /opt/ros/melodic/setup.bash
catkin_make -C ~/catkin_ws
source ~/catkin_ws/devel/setup.bash
source /opt/ros/melodic/setup.bash
source /usr/share/gazebo/setup.sh



source /opt/ros/melodic/setup.bash
source ~/catkin_build_ws/install/setup.bash



source /opt/ros/melodic/setup.bash
source ~/catkin_build_ws/install/setup.bash --extend
source /opt/ros/melodic/setup.bash
source /usr/share/gazebo/setup.sh
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws/src


source ~/catkin_build_ws/install/setup.bash --extend
source /usr/share/gazebo/setup.sh
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws/src
"""