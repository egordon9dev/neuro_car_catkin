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
import msgpack
import time
bridge = CvBridge()

def mag(x):
    return (x[0]**2 + x[1]**2) ** 0.5
def downscale_action(x):
    return np.array([x[0]/4, x[1]/1.5])
def upscale_action(x):
    return np.array([x[0]*4, x[1]*1.5])

def laser_callback(laser_scan):
    pass

width = 640
height = 360

shrink_width = 128
shrink_height = 72

img_arr = np.ones(shrink_width*shrink_height)
log_msg = String()
real_twist = Twist()

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

    # blown_up = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("Image window", blown_up)
    # cv2.waitKey(3)

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

last_log = 0
fail_load_weights = False
its = 0
def move(t, x):
    global last_log, fail_load_weights, its

    max_speed = 2
    max_angular = 0.5

    its += 1
    action = upscale_action(x)
    trans_vel = action[0]
    angular_vel = action[1]
    angular_vel = max(min(angular_vel, max_angular), -max_angular)
    trans_vel = max(min(trans_vel, max_speed), -max_speed)
    

    neurocar_msg.linear.x = trans_vel
    neurocar_msg.angular.z = angular_vel
    pub.publish(neurocar_msg)
    if t-last_log > 0.3:
        pub_log.publish(log_msg)
        last_log = t

    # do action
    rate.sleep()


    log_msg.data = "Failed to Load Weights" if fail_load_weights else "t: " + str(t) + " vel: (" + str(real_twist.linear.x) + ", " + str(real_twist.angular.z) + ") %d "%(its)
    return x

def get_next_data(t):
    global img_arr
    return img_arr

class Explicit(nengo.solvers.Solver):
    def __init__(self, value, weights=False):
        super(Explicit, self).__init__(weights=weights)
        self.value = value
            
    def __call__(self, A, Y, rng=None, E=None):
        return self.value, {}
n_stim_neurons = 8000
model = nengo.Network(seed=8)
with model:
    movement = nengo.Ensemble(n_neurons=8000, dimensions=2)

    movement_node = nengo.Node(move, size_in=2, size_out=2, label='Movement')
    nengo.Connection(movement, movement_node)

    stim_ensemble = nengo.Ensemble(n_neurons=n_stim_neurons, dimensions=9216)
    stim_camera = nengo.Node(get_next_data)
    nengo.Connection(stim_camera, stim_ensemble)
    try:
        weights_trans = np.load("/home/ethan/catkin_ws/src/neurocar/src/weights_trans.npy")
    except IOError:
        weights_trans = np.zeros((n_stim_neurons, 1))
        fail_load_weights = True
    try:
        weights_rot = np.load("/home/ethan/catkin_ws/src/neurocar/src/weights_rot.npy")
    except IOError:
        weights_rot = np.zeros((n_stim_neurons, 1))
        fail_load_weights = True

    conn_trans = nengo.Connection(stim_ensemble, movement, function=lambda x: 0.7, transform=[[1], [0]], solver = Explicit(weights_trans))
    conn_rot = nengo.Connection(stim_ensemble, movement, function=lambda x: 0, transform=[[0], [1]], solver = Explicit(weights_rot))

simulator = nengo.Simulator(model)
simulator.run(30)

def main():
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
