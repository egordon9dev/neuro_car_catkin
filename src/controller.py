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
import nengo

bridge = CvBridge()

width = 1280
height = 720
angular_vel = 0
trans_vel = 0

def dist(p1, p2):
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5;


def find_curbs(contours):
    if len(contours) < 2:
        print("Error finding curbs: len less than 2")

    p1 = None
    p2 = None
    contour1 = None
    contour2 = None
    p1_est = (width/2, 0)
    p2_est = (width/2, 0)
    for contour in contours:
        m = cv2.moments(contour)
        if(m["m00"] == 0):
            continue
        cx = m["m10"]/m["m00"]
        cy = m["m01"]/m["m00"]
        cur = (cx, cy)
        if p1 is None or dist(p1, p1_est) > dist(cur, p1_est):
            contour1 = contour
        elif p2 is None or dist(p2, p2_est) > dist(cur, p2_est):
            contour2 = contour
    return (contour1, contour2)


def find_center_point(contour1, contour2):
    m1 = cv2.moments(contour1)
    m2 = cv2.moments(contour2)
    p1 = (m1["m10"]/m1["m00"], m1["m01"]/m1["m00"])
    p2 = (m2["m10"]/m2["m00"], m2["m01"]/m2["m00"])
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def move(t, x):
    speed, rotation = x


    #return reward

def callback(img):
    global angular_vel, trans_vel
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    edges_img = cv2.Canny(cv_image, 50,250)
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
        if(mom["m00"] == 0):
            continue
        center_point = (mom["m10"]/mom["m00"], mom["m01"]/mom["m00"])
        reward += dist(car_point, center_point) + trans_vel

        # [vx1, vy1, x1, y1] = cv2.fitLine(curbs[0], cv2.DIST_L2,0,0,0.01,0.01)

    ndarray = np.asarray(cv_image)

    cv2.imshow("Image window", contours_img)
    cv2.waitKey(3)


log_string = String()


def laserCallback(laserScan):
    global log_string
    log_string.data = "got laser: " + str(laserScan.range_min) + " to " + str(laserScan.range_max);


def main():
    global angular_vel
    global log_string
    sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
    ray_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laserCallback)
    pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
    log_pub = rospy.Publisher('neurocar/logger', String, queue_size=10)
    rospy.init_node('controller', anonymous=True)
    rate = rospy.Rate(2) # 10hz
    msg = Twist()
    msg.linear.x = 0
    while not rospy.is_shutdown():
        msg.angular.z = angular_vel
        pub.publish(msg)
        log_pub.publish(log_string)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
