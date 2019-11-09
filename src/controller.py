#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np
import math
import nengo
import nengo_fpga
from nengo_fpga.networks import FpgaPesEnsembleNetwork

bridge = CvBridge()

width = 1280
height = 720
angular_vel = 0
trans_vel = 0
img_arr = None

# acceleration: translational, rotational
actions_list = [
    [[-1], [-1]],
    [[-1], [0]],
    [[-1], [1]],
    [[0], [-1]],
    [[0], [0]],
    [[0], [1]],
    [[1], [-1]],
    [[1], [0]],
    [[1], [1]]
]


def dist(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5;


def find_curbs(contours):
    if len(contours) < 2:
        print("Error finding curbs: len less than 2")

    p1 = None
    p2 = None
    contour1 = None
    contour2 = None
    p1_est = (width / 2, 0)
    p2_est = (width / 2, 0)
    for contour in contours:
        m = cv2.moments(contour)
        if (m["m00"] == 0):
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        cur = (cx, cy)
        if p1 is None or dist(p1, p1_est) > dist(cur, p1_est):
            contour1 = contour
        elif p2 is None or dist(p2, p2_est) > dist(cur, p2_est):
            contour2 = contour
    return (contour1, contour2)


def find_center_point(contour1, contour2):
    m1 = cv2.moments(contour1)
    m2 = cv2.moments(contour2)
    p1 = (m1["m10"] / m1["m00"], m1["m01"] / m1["m00"])
    p2 = (m2["m10"] / m2["m00"], m2["m01"] / m2["m00"])
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def callback(img):
    global angular_vel, trans_vel
    global img_arr
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    edges_img = cv2.Canny(cv_image, 50, 250)
    contours_img, contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    # src = np.float32([(0, 180), (1280, 180), (700, 360), (580, 360)])
    # dst = np.float32([(0,180), (1280, 180), (1280, 360), (0, 360)])
    # M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    # warped = cv2.warpPerspective(cv_image, M, (1280, 720))
    # track = findCurbs(contours)
    # curbs = find_curbs(contours)
    # cv2.drawContours(contours_img, curbs, -1, (0,255,0), 3)
    #
    # avg_vec = (1,0)
    # mid_x = width/2
    # mid_y = height/2
    reward = 0
    car_point = (640, 100)
    for cont in contours:
        mom = cv2.moments(cont)
        if (mom["m00"] == 0):
            continue
        center_point = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])
        reward += dist(car_point, center_point) + trans_vel

        # [vx1, vy1, x1, y1] = cv2.fitLine(curbs[0], cv2.DIST_L2,0,0,0.01,0.01)

    img_arr = np.asarray(contours_img).flatten()

    cv2.imshow("Image window", contours_img)
    cv2.waitKey(3)


log_string = String()
log_string.data = ""
min_range = 10000
min_range_angle = 0
max_range = 0
max_range_angle = 0


def laser_callback(laser_scan):
    global min_range, min_range_angle, max_range, max_range_angle
    global log_string
    avg_range = 0
    min_range = 100000
    min_range_angle = 0
    max_range = 0
    max_range_angle = 0
    if len(laser_scan.ranges) > 0:
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


sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
ray_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laser_callback)
pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
log_pub = rospy.Publisher('neurocar/logger', String, queue_size=10)
rospy.init_node('controller', anonymous=True)
rate = rospy.Rate(60)  # 10hz
neurocar_msg = Twist()


def polar_to_rect(r, theta):
    return r * math.cos(theta), r * math.sin(theta)


#
# def main():
#     global angular_vel, trans_vel
#     global log_string
#     while not rospy.is_shutdown():


lastAction = (0.0, 0.0)


def move(t, x):
    global angular_vel, trans_vel

    prev_min_range = min_range

    acceleration = [0.0, 0.0]
    for i in range(len(x)):
        action = actions_list[i]
        intensity = x[i]
        acceleration[0] += intensity * action[0][0]
        acceleration[1] += intensity * action[1][0]
    trans_vel += acceleration[0]
    angular_vel += acceleration[1]

    # send action
    neurocar_msg.linear.x = trans_vel
    neurocar_msg.angular.z = angular_vel
    pub.publish(neurocar_msg)

    if len(log_string.data) > 0:
        log_pub.publish(log_string)
        log_string.data = ""

    # do action
    rate.sleep()

    # network reward
    # min_range_vec = polar_to_rect(min_range, min_range_angle)
    # max_range_vec = polar_to_rect(max_range, max_range_angle)
    # delta_vec = (max_range_vec[0] - min_range_vec[0], max_range_vec[1] - min_range_vec[1])

    reward = 0
    # move away from obstacles
    reward += min_range - prev_min_range
    # go fast
    reward += 10 * abs(neurocar_msg.linear.x)

    return reward


class InputManager:
    """
    The InputManager maintains the current input data for a neural network.
    To update the data, please call fn and ensure that
    the data dimensionality is exactly the same as the previous data.
    """

    def __init__(self, input_function, dimensions):
        self.function = input_function
        self.dimensions = dimensions


class NeuralNet:
    """
    A NeuralNet contains a Nengo network with an FPGA implementation.
    It intends to receive data through an InputManager.
    It will select one of a few actions and execute a given function.
    """

    def __init__(self, input_manager, act_function, learning_active=1, board="pynq", learn_rate=1e-4,
                 learn_synapse=0.030, action_threshold=0.1, init_transform=[1, 1, 1, 1, 1, 1, 1, 1, 1]):
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
            # Set up the input
            input_node = nengo.Node(self.input_manager.function)

            # Set up the movement node
            movement = nengo.Ensemble(n_neurons=100, dimensions=2)
            movement_node = nengo.Node(act_function, size_in=2, label="reward")
            nengo.Connection(movement, movement_node)

            # Create the action selection networks
            basal_ganglia = nengo.networks.actionselection.BasalGanglia(len(actions_list))
            thalamus = nengo.networks.actionselection.Thalamus(len(actions_list))
            nengo.Connection(basal_ganglia.output, thalamus.input)

            # Convert the selection actions to act transforms

            for i, action in enumerate(actions_list):
                nengo.Connection(thalamus.output[i], movement, transform=action)

            # Generate the training (error) signal
            def error_func(t, x):
                actions = np.array(x[:9])
                utils = np.array(x[9:18])
                r = x[19]
                activate = x[20]

                max_action = max(actions)
                actions[actions < self.action_threshold] = 0
                actions[actions != max_action] = 0
                actions[actions == max_action] = 1

                return activate * (
                        np.multiply(actions, (utils - r) * (1 - r) ** 5)
                        + np.multiply((1 - actions), (utils - 1) * (1 - r) ** 5)
                )

            errors = nengo.Node(error_func, size_in=21, size_out=9)
            nengo.Connection(thalamus.output, errors[:9])
            nengo.Connection(basal_ganglia.input, errors[9:18])
            nengo.Connection(movement_node, errors[19])

            # Learning on the FPGA
            adaptive_ensemble = FpgaPesEnsembleNetwork(
                self.board,
                n_neurons=1000,
                dimensions=self.input_manager.dimensions,
                learning_rate=self.learn_rate,
                function=lambda x: self.init_transform,
                label="pes ensemble"
            )

            nengo.Connection(input_node, adaptive_ensemble.input, synapse=self.learn_synapse)
            nengo.Connection(errors, adaptive_ensemble.error)
            nengo.Connection(adaptive_ensemble.output, basal_ganglia.input)

            learn_on = nengo.Node(self.learning_active)
            nengo.Connection(learn_on, errors[20])

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
    global width, height
    input_manager = InputManager(lambda t: img_arr, width * height)
    network = NeuralNet(input_manager, move)
    network.run_network(90)
    network.close_simulator()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
