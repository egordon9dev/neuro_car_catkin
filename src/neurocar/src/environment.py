import gym
from gym import spaces
import numpy as np
from geometry_msgs.msg import Twist, Pose, ModelState
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin, cos
import time
import random
from scipy.spatial.transform import Rotation
import rospy

bridge = CvBridge()
pub_action = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
pub_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
neurocar_msg = Twist()
rate = rospy.Rate(60)

shrink_height = 36
shrink_width = 64

observation_ready = False
def reset_observation():
    observation_ready = False
    observation = None
    reward = None
    real_twist = None
    ranges = None
def await_observation():
    while(observation_ready == False):
        rospy.sleep(rospy.Duration(nsecs=2000))

display_img = None
def img_callback(img):
    try:
        img = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    img = cv2.resize(img, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)
    display_img = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    await_observation = False

def laser_callback(laser_scan):
    ranges = laser_scan.ranges

def odom_callback(odom):
    real_twist = odom.twist.twist

img_sub = rospy.Subscriber('neurocar/camera/image_raw', Image, img_callback)
laser_sub = rospy.Subscriber('neurocar/laser/scan', LaserScan, laser_callback)
odom_sub = rospy.Subscriber('neurocar/odom', Odometry, odom_callback)

actions_list = [
    { "linear": 1, "angular": -1 },
    { "linear": 1, "angular": 0 },
    { "linear": 1, "angular": 1 }
]
img_shape = (36, 64, 3)

def get_random_state():
    result = ModelState()
    result.model_name = "neurocar"
    rand1 = random.random()
    if rand1 < .5:
        # straightaway
        y0 = -7.2 if random.random() < .5 else 5.1
        x0 = -6
        result.pose.position.x = x0 + random.random() * 12
        result.pose.position.y = y0 + random.random() * 2.1
    else:
        #turn
        istop = random.random() > .5
        y0 = 0
        x0 = 6 if istop else -6
        r = 5.2 + 1.9 * random.random()
        theta = random.random() * 3.14
        if istop:
            theta -= 3.14 / 2
        else:
            theta += 3.14 / 2
        result.pose.position.x = x0 + r * cos(theta)
        result.pose.position.y = y0 + r * sin(theta)
    quaternion = Rotation.from_euler("z", random.random()*360, degrees=True).as_quat()
    result.pose.orientation.x = quaternion[0]
    result.pose.orientation.y = quaternion[1]
    result.pose.orientation.z = quaternion[2]
    result.pose.orientation.w = quaternion[3]
    result.twist = Twist()
    return result

class NeurocarEnv(gym.Env):
    """Custom Neurocar Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(NeurocarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_list = actions_list
        self.action_space = spaces.Discrete(len(actions_list))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        reset_observation()

    def step(self, action):
        pub_action.publish(Twist(**self.actions_list[action]))
        await_observation()
        obs = observation
        rew = reward
        done = False
        info = None
        reset_observation()
        return obs, rew, done, info

    def reset(self):
        pub_state.publish(get_random_state())
        reset_observation()
        await_observation()
        obs = observation
        reset_observation()
        return obs

    def render(self, mode='human'):
        cv2.imshow("Image window", display_img)
        cv2.waitKey(3)

    def close (self):
        pass