#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np
import math
import keras
import nengo

bridge = CvBridge()

width = 1280
height = 720
shrink_width = 256
shrink_height = 144
angular_vel = 0
trans_vel = 0
rospy.init_node('controller', anonymous=True)

pub = rospy.Publisher('/neurocar/cmd_vel', Twist, queue_size=10)
logger = rospy.Publisher("/neurocar/log", String, queue_size=1)

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

def image_callback(img):
    logger.publish(str(img.shape))
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
sub = rospy.Subscriber('/neurocar/camera/image_raw', Image, image_callback)

# print("Syntax correct.")
def main():
    logger.publish("starting main loop.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
