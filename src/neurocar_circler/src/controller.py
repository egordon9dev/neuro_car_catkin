#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

from math import sin

stay_stil = True

def main():
    pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
    rospy.init_node('neurocar_circler', anonymous=True)

    rate = rospy.Rate(2) # 10hz
    msg = Twist()
    msg.linear.x = 0
    msg.angular.z = 3

    while not rospy.is_shutdown():
        msg.linear.x += .2
        if stay_still: pub.publish(Twist())
        else: pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
