#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

pub = rospy.Publisher("chatter", String, queue_size=10)
rospy.init_node("chatter")
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    msg = String()
    msg.data = "hello"
    pub.publish(msg)
    rate.sleep()
