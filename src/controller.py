#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from math import sin
gotData = False
n = 0
def callback(img):
    global n
    global gotData
    gotData = True
    file = open("out", "wb")
    n+=1
    file.write("QWERTY " + str(n) + "\n")
    file.close()

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
        if gotData:
            pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
