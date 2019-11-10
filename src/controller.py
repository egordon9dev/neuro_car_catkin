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
lasers = 100 * np.ones(3)
prev_lasers = 100 * np.ones(3)


def laser_callback(laser_scan):
    global min_range, min_range_angle, max_range, max_range_angle, lasers
    avg_range = 0
    min_range = 100000
    min_range_angle = 0
    max_range = 0
    max_range_angle = 0
    prev_lasers = lasers
    if len(laser_scan.ranges) > 0:
        lasers = [max(min(100, rng), 0) for rng in laser_scan.ranges]
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

shrink_width = 256
shrink_height = 144

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

# sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
ray_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laser_callback)
odom_sub = rospy.Subscriber('neurocar/odom', Odometry, odom_callback)
pub_log = rospy.Publisher('neurocar/log', String, queue_size=10)
pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
rospy.init_node('controller', anonymous=True)
rate = rospy.Rate(60)  # 10hz
neurocar_msg = Twist()

last_log = 0
def move(t, x):
    global angular_vel, trans_vel, min_range
    global lasers, prev_lasers, last_log
    reward = 0
    #
    # max_speed = 10
    # max_angular = 0.05
    #
    trans_vel = x[0] * 50
    angular_vel = x[1] * 0.01
    # angular_vel = max(min(angular_vel, max_angular), -max_angular)
    # trans_vel = max(min(trans_vel, max_speed), -max_speed)
    min_laser = min(lasers)
    max_laser = max(lasers)
    if min_laser < 1:
        if trans_vel > 0:
            neurocar_msg.linear.x = 0
            neurocar_msg.angular.z = 0
            pub.publish(neurocar_msg)
            return 0
    else:
        if trans_vel < 0:
            neurocar_msg.linear.x = 0
            neurocar_msg.angular.z = 0
            pub.publish(neurocar_msg)
            return 0

    prev_min_laser = min_laser
    prev_mid_laser = lasers[1]
    # send action
    neurocar_msg.linear.x = max(min(neurocar_msg.linear.x + trans_vel, 5), -5)
    neurocar_msg.angular.z = angular_vel
    # neurocar_msg.angular.z = 0 if real_twist.angular.z > 1 else angular_vel
    pub.publish(neurocar_msg)
    if t-last_log > 0.3:
        pub_log.publish(log_msg)
        last_log = t

    # do action
    rate.sleep()

    # move away from obstacles

    delta_lasers = np.absolute(np.subtract(lasers, prev_lasers))
    if math.isfinite(min_laser) and min_laser > 5 or min_laser > prev_min_laser:
        reward += min_laser * 10
    #
    #
    if min_laser > 2:
        reward += (real_twist.linear.x ** 2) * 300
    if min_laser > 3:
        reward += (real_twist.linear.x ** 2) * 900
    if min_laser > 5:
        reward += (real_twist.linear.x ** 2) * 1500
    if abs(angular_vel) < 0.01:
        reward += 3
    if real_twist.linear.x >= 0:
        if lasers[1] > prev_mid_laser:
            reward += 100
    if lasers[1] > 4:
        reward += 300 + real_twist.linear.x * 100
    #
    log_msg.data = "t: " + str(t) + " reward: " + str(reward) + "\nmax lsr: " + str(max(lasers)) + " min lsr: " + str(min(lasers)) + " vel: (" + str(real_twist.linear.x) + ", " + str(real_twist.angular.z) + ")\n"
    return reward


class InputManager:
    """
    The InputManager maintains the current input data for a neural network.
    To update the data, please call fn and ensure that
    the data dimensionality is exactly the same as the previous data.
    """

    def __init__(self):
        global lasers
        self.dimensions = len(lasers)

    def function(self, t):
        global lasers
        return lasers

class NeuralNet:
    """
    A NeuralNet contains a Nengo network with an FPGA implementation.
    It intends to receive data through an InputManager.
    It will select one of a few actions and execute a given function.
    """

    def __init__(self, input_manager, act_function, learning_active=1, board="pynq", learn_rate=1e-4,
                 learn_synapse=0.030, action_threshold=0.1, init_transform=[0, 0, 0, 1]):
        global actions_list, lasers
        self.model = nengo.Network(seed=8)
        self.input_manager = input_manager
        self.learning_active = learning_active
        self.board = board
        # parameters for the learning model (i.e. i don't know what they really do)
        self.learn_rate = learn_rate
        self.learn_synapse = learn_synapse
        self.action_threshold = action_threshold
        self.init_transform = init_transform

        with self.model:
            movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)

            movement_node = nengo.Node(move, size_in=2, label='reward')
            nengo.Connection(movement, movement_node)

            radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4)
            stim_radar = nengo.Node(lambda t: lasers)
            nengo.Connection(stim_radar, radar)

            bg = nengo.networks.actionselection.BasalGanglia(3)
            thal = nengo.networks.actionselection.Thalamus(3)
            nengo.Connection(bg.output, thal.input)

            def u_fwd(x):
                return 0.8

            def u_left(x):
                return 0.6

            def u_right(x):
                return 0.7

            conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES())
            conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES())
            conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES())

            nengo.Connection(thal.output[0], movement, transform=[[1], [0]])
            nengo.Connection(thal.output[1], movement, transform=[[0], [1]])
            nengo.Connection(thal.output[2], movement, transform=[[0], [-1]])

            errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=3)
            nengo.Connection(movement_node, errors.input, transform=-np.ones((3, 1)))
            nengo.Connection(bg.output[0], errors.ensembles[0].neurons, transform=np.ones((50, 1)) * 4)
            nengo.Connection(bg.output[1], errors.ensembles[1].neurons, transform=np.ones((50, 1)) * 4)
            nengo.Connection(bg.output[2], errors.ensembles[2].neurons, transform=np.ones((50, 1)) * 4)
            nengo.Connection(bg.input, errors.input, transform=1)

            nengo.Connection(errors.ensembles[0], conn_fwd.learning_rule)
            nengo.Connection(errors.ensembles[1], conn_left.learning_rule)
            nengo.Connection(errors.ensembles[2], conn_right.learning_rule)

        self.simulator = nengo.Simulator(self.model)

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
    network.run_network(600)
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