#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np

bridge = CvBridge()

def callback(img):
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    ndarray = np.asarray(cv_image)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def main():
    sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
    pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
    rospy.init_node('controller', anonymous=True)

    rate = rospy.Rate(2) # 10hz
    msg = Twist()
    msg.linear.x = 0
    msg.angular.z = 3

    msg.linear.x = 5
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
