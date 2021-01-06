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
    def __init__(self, classifier_param_file:str, log_inputs=False):
        self.color_classifier = CarpetColorClassifier(classifier_param_file)

        # load map
        # for now, using a randomly generated map
        # TODO: map saving/loading
        shape = (40, 40)
        cell_size = 0.5
        n_colors = 4
        np.random.seed(123)
        from cbl_particle_filter.carpet_map import generate_random_map
        carpet = generate_random_map(shape, cell_size, n_colors)

        self.particle_filter = CarpetBasedParticleFilter(carpet)
        self.cv_bridge = CvBridge()
        self.log_inputs = log_inputs
        self.input_log = []
        self.previous_odom = None

    def __del__(self):
        if self.log_inputs:
            log_path = '/tmp/carpet_localisation_inputs.pickle'
            with open(log_path, 'wb') as f:
                pickle.dump(self.input_log, f, protocol=pickle.HIGHEST_PROTOCOL)
            rospy.loginfo(f"Saved input log to '{log_path}'")

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

        if self.log_inputs:
            self.input_log.append((odom_delta, color_measurement))


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

    classifier_param_file = rospy.get_param("~classifier_param_file")

    # param 'log_inputs': set true to record all inputs to the particle filter
    # as a pickle file, for later offline playback
    log_inputs = rospy.get_param("~log_inputs", default=False)

    localiser = CarpetLocaliser(classifier_param_file, log_inputs)

    odom_sub = message_filters.Subscriber('odom', Odometry)
    penalty_sub = message_filters.Subscriber('image', Image)

    time_synchronizer = message_filters.ApproximateTimeSynchronizer([odom_sub, penalty_sub], 10, 0.2)
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



# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass