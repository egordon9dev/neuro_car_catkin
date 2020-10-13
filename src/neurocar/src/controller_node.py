#! /usr/bin/env python
import rospy
from std_msgs.msg import String

with open("test999", "w") as f:
    f.write("this is only a test")

pub = rospy.Publisher("/neurocar/log", String, queue_size=1)
rospy.init_node("controller_node", anonymous=True)
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rospy.loginfo("hello from ctrl")
    print("test")
    pub.publish("test")
    rate.sleep()

pub.publish("controller_node died")
print("test")