#! /usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(s):
    print("recieved: " + s.data)
sub = rospy.Subscriber("chatter", String, callback)
rospy.init_node("chatter_sub")
rate = rospy.Rate(10)
rospy.spin()