#!/usr/bin/env python
from typing import Tuple
import pickle
import numpy as np
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters
from tf.transformations import euler_from_quaternion
from carpet_color_classification import CarpetColorClassifier
from cbl_particle_filter.filter import CarpetBasedParticleFilter, Pose, OdomMeasurement, ColorMeasurement
from cbl_particle_filter.carpet_map import load_map_from_png

def odom_msg_to_measurement(odom_msg:Odometry) -> OdomMeasurement:
    """
    Convert ROS odom to the cbl representation
    """
    q = odom_msg.pose.pose.orientation
    _, _, heading = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return OdomMeasurement(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        heading,
    )

def compute_odom_delta(current_odom:OdomMeasurement, previous_odom:OdomMeasurement) -> OdomMeasurement:
    """
    return pose delta from prev to current, in prev frame
    """
    if previous_odom is None:
        return OdomMeasurement(0,0,0)

    dx_global = current_odom.dx - previous_odom.dx
    dy_global = current_odom.dy - previous_odom.dy
    prev_heading = previous_odom.dheading

    return OdomMeasurement(
        dx =  dx_global * np.cos(prev_heading) + dy_global * np.sin(prev_heading),
        dy = -dx_global * np.sin(prev_heading) + dy_global * np.cos(prev_heading),
        dheading= current_odom.dheading - previous_odom.dheading
    )


class CarpetLocaliser():
    """
    Interface between incoming ROS messages (odom and camera image) and
    carpet based particle filter.
    """
    def __init__(
            self,
            map_png_file:str,
            map_cell_size:float,
            classifier_param_file:str,
            log_inputs=False):

        self.color_classifier = CarpetColorClassifier(classifier_param_file)
        self.log_inputs = log_inputs

        carpet = load_map_from_png(map_png_file, map_cell_size)

        self.particle_filter = CarpetBasedParticleFilter(carpet, log_inputs)
        self.cv_bridge = CvBridge()
        self.previous_odom = None

    def __del__(self):
        if self.log_inputs:
            log_path = "/tmp/localiser_input_log.pickle"
            self.particle_filter.write_input_log(log_path)  
            rospy.loginfo(f"Saved localiser input log to {log_path}")


    def localisation_update(self, odom_msg:Odometry, img_msg:Image):
        """
        Update localisation based on given odom,image pair
        """
        rospy.loginfo("callback")

        # determine detected color
        color_measurement, color_name = self._classify_image_color(img_msg)
        rospy.loginfo(f"detected color: {color_name}")

        # determine odom delta since last update
        current_odom = odom_msg_to_measurement(odom_msg)
        odom_delta = compute_odom_delta(current_odom, self.previous_odom)
        rospy.loginfo(f"odom: {odom_delta}")
        self.previous_odom = current_odom

        self.particle_filter.update(odom_delta, color_measurement)



    def _classify_image_color(self, img_msg:Image) -> Tuple[ColorMeasurement, str]:
        """
        invoke classifier on given image, returning color index and name (string)
        """
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        color_index, color_name = self.color_classifier.classify(cv_image)

        # set index to None if classification failed
        if color_name == "UNCLASSIFIED":
            color_index = None
        return ColorMeasurement(color_index), color_name


def run_localisation():
    rospy.init_node("carpet_localisation")
    rospy.loginfo("initialised")

    map_png_file = rospy.get_param("~map_png_file")
    map_cell_size = rospy.get_param("~map_cell_size")
    classifier_param_file = rospy.get_param("~classifier_param_file")

    # param 'log_inputs': set true to record all inputs to the particle filter
    # as a pickle file, for later offline playback
    log_inputs = rospy.get_param("~log_inputs", default=False)

    localiser = CarpetLocaliser(map_png_file, map_cell_size, classifier_param_file, log_inputs)

    odom_sub = message_filters.Subscriber('odom', Odometry)
    image_sub = message_filters.Subscriber('image', Image)

    time_synchronizer = message_filters.ApproximateTimeSynchronizer([odom_sub, image_sub], 10, 0.2)
    time_synchronizer.registerCallback(localiser.localisation_update)


    def img_callback(img_msg:Image):
        cv_image = cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        color_index, color_name = classifier.classify(cv_image)
        rospy.loginfo(f"detected color: {color_name}")
        
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("INTERRUPT!")

    del localiser
    print("DONE!")
    

if __name__ == '__main__':
    run_localisation()
