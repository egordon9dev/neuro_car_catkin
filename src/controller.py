#!/usr/bin/env python3
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

def callback(img):
    pass
    # global angular_vel, trans_vel
    # try:
    #     cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    # except CvBridgeError as e:
    #     print(e)
    # edges_img = cv2.Canny(cv_image, 50, 250)
    # contours_img, contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    #
    # cv2.imshow("Image window", contours_img)
    # cv2.waitKey(3)

sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
rospy.init_node('controller', anonymous=True)
rate = rospy.Rate(60)  # 10hz
neurocar_msg = Twist()

def move(t, x):
    global angular_vel, trans_vel

    trans_vel += x[0]
    angular_vel += x[1]
    trans_vel = max(min(trans_vel, 2), -2)
    angular_vel = max(min(angular_vel, 2), -2)


    # send action
    neurocar_msg.linear.x = trans_vel
    neurocar_msg.angular.z = angular_vel
    pub.publish(neurocar_msg)

    # do action
    rate.sleep()

    return trans_vel * 100 + angular_vel * 10


class InputManager:
    """
    The InputManager maintains the current input data for a neural network.
    To update the data, please call fn and ensure that
    the data dimensionality is exactly the same as the previous data.
    """

    def __init__(self):
        self.dimensions = 1

    def function(self, t):
        return np.array([1])

class NeuralNet:
    """
    A NeuralNet contains a Nengo network with an FPGA implementation.
    It intends to receive data through an InputManager.
    It will select one of a few actions and execute a given function.
    """

    def __init__(self, input_manager, act_function, learning_active=1, board="pynq", learn_rate=1e-5,
                 learn_synapse=0.030, action_threshold=0.1, init_transform=[0, 0, 0, 0, 0, 0, 0, 1, 0]):
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
            # stim_ensemble = nengo.Ensemble(n_neurons=50, dimensions=self.input_manager.dimensions)
            stim_ensemble_fpga = FpgaPesEnsembleNetwork(
                self.board,
                n_neurons=50,
                dimensions=self.input_manager.dimensions,
                learning_rate=self.learn_rate
            )
            nengo.Connection(stim_node, stim_ensemble_fpga.input)

            # Create the action selection networks
            basal_ganglia = nengo.networks.actionselection.BasalGanglia(len(actions_list))
            thalamus = nengo.networks.actionselection.Thalamus(len(actions_list))
            nengo.Connection(basal_ganglia.output, thalamus.input)

            # Convert the selection actions to act transforms
            for i, action in enumerate(actions_list):
                nengo.Connection(stim_ensemble_fpga.output, basal_ganglia.input[i])
                nengo.Connection(thalamus.output[i], movement, transform=action)

            errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=len(actions_list))
            nengo.Connection(movement_node, errors.input, transform=-np.ones((len(actions_list),1)))
            for i in range(len(actions_list)):
                nengo.Connection(basal_ganglia.output[i], errors.ensembles[i].neurons, transform=np.ones((50, 1)) * 4)
            nengo.Connection(basal_ganglia.input, errors.input, transform=1)
            for i in range(len(actions_list)):
                nengo.Connection(errors.ensembles[i], stim_ensemble_fpga.error)

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
    global width, height, img_arr
    input_manager = InputManager()
    network = NeuralNet(input_manager, move)
    network.run_network(60)
    network.close_simulator()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
