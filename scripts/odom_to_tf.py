#!/usr/bin/env python

# util for publishing odom over tf, based on subscription to odom topic
# used for republishing odom during log playback

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf_conversions
import tf2_ros


def odom_callback(odom_msg: Odometry, br:tf2_ros.TransformBroadcaster):
    t = TransformStamped()

    t.header = odom_msg.header
    t.child_frame_id = "base_link"
    t.transform.translation.x = odom_msg.pose.pose.position.x
    t.transform.translation.y = odom_msg.pose.pose.position.y
    t.transform.translation.z = odom_msg.pose.pose.position.z
    t.transform.rotation = odom_msg.pose.pose.orientation

    br.sendTransform(t)

if __name__ == '__main__':

    rospy.init_node('odom_to_tf')
    br = tf2_ros.TransformBroadcaster()
    rospy.Subscriber('odom',
                     Odometry,
                     odom_callback,
                     br)
    rospy.spin()