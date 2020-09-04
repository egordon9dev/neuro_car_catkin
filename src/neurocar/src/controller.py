#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np
import math
import nengo
import keras
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


def image_callback(img):
    global img_arr, width, height
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    cv_image = cv2.resize(cv_image, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)
    # # edges_img = cv2.Canny(cv_image, 50, 250)
    # hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # lower = np.array([0,0,0])
    # upper = np.array([255,255,10])
    # thresh = cv2.inRange(hsv, lower, upper)
    img_arr = np.asarray(cv_image).astype(np.uint8)
    print(str(img_arr.shape) + " max: {0} min: {1}"%(max(img_arr), min(img_arr)))

    blown_up = cv2.resize(img_arr, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image window", blown_up)
    cv2.waitKey(3)


def odom_callback(odom):
    global real_twist
    real_twist = odom.twist.twist


sub = rospy.Subscriber('neurocar/camera/image_raw', Image, image_callback)
ray_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laser_callback)
odom_sub = rospy.Subscriber('neurocar/odom', Odometry, odom_callback)
pub_log = rospy.Publisher('neurocar/log', String, queue_size=10)
pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
rospy.init_node('controller', anonymous=True)
neurocar_msg = Twist()

last_log = 0
fail_load_weights = False
its = 0
def move(t, x):
    # calculate translational and rotational velocity
    global last_log, fail_load_weights, its
    max_speed = 4.4
    max_angular = 2.2
    its += 1
    action = upscale_action(x)
    trans_vel = np.clip(action[0], -max_speed, max_speed)
    angular_vel = np.clip(action[1], -max_angular, max_angular)

    # publish velocity command
    neurocar_msg.linear.x = trans_vel
    neurocar_msg.angular.z = angular_vel
    pub.publish(neurocar_msg)

    # publish log data
    log_msg.data = "Failed to Load Weights" if fail_load_weights else "t: " + str(t) + " vel: (" + str(real_twist.linear.x) + ", " + str(real_twist.angular.z) + ") %d "%(its)
    if t-last_log > 0.3:
        pub_log.publish(log_msg)
        last_log = t
    
    return x


def main():
    rospy.loginfo("Running main loop")
    pub_log.publish("Running main loop")
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rospy.loginfo("loop")
        pub_log.publish("loop")
        move(0, [0.1, 0])
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
