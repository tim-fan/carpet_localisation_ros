#!/usr/bin/env python
from typing import Tuple, Optional
import pickle
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion, PoseArray
from geometry_msgs.msg import Pose as PoseMsg
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from carpet_color_classification import CarpetColorClassifier
from cbl_particle_filter.filter import CarpetBasedParticleFilter, Pose, OdomMeasurement, ColorMeasurement
from cbl_particle_filter.carpet_map import load_map_from_png


def yaw_from_quaternion_msg(q: Quaternion) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def quaternion_msg_from_yaw(yaw: float) -> Quaternion:
    q_list = quaternion_from_euler(0, 0, yaw)
    q = Quaternion(
        x=q_list[0],
        y=q_list[1],
        z=q_list[2],
        w=q_list[3],
    )
    return q


def odom_msg_to_measurement(odom_msg: Odometry) -> OdomMeasurement:
    """
    Convert ROS odom to the cbl representation
    """
    heading = yaw_from_quaternion_msg(odom_msg.pose.pose.orientation)
    return OdomMeasurement(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        heading,
    )


def odom_msg_to_pose(odom_msg: Odometry) -> Pose:
    """
    Convert ROS odom to the cbl pose representation
    """
    heading = yaw_from_quaternion_msg(odom_msg.pose.pose.orientation)
    return Pose(
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        heading,
    )


def pose_to_pose_msg(pose: Pose) -> PoseMsg:
    """
    convert cbl pose to ROS pose message
    """
    pose_msg = PoseMsg()
    pose_msg.position.x = pose.x
    pose_msg.position.y = pose.y
    pose_msg.orientation = quaternion_msg_from_yaw(pose.heading)
    return pose_msg


def wrap_plus_minus_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def compute_odom_delta(current_odom: OdomMeasurement,
                       previous_odom: OdomMeasurement) -> OdomMeasurement:
    """
    return pose delta from prev to current, in prev frame
    """
    if previous_odom is None:
        return OdomMeasurement(0, 0, 0)

    dx_global = current_odom.dx - previous_odom.dx
    dy_global = current_odom.dy - previous_odom.dy
    prev_heading = previous_odom.dheading

    return OdomMeasurement(dx=dx_global * np.cos(prev_heading) +
                           dy_global * np.sin(prev_heading),
                           dy=-dx_global * np.sin(prev_heading) +
                           dy_global * np.cos(prev_heading),
                           dheading=wrap_plus_minus_pi(current_odom.dheading -
                                                       previous_odom.dheading))


def particles_to_pose_array(particles: np.ndarray) -> PoseArray:
    """
    Convert particle filter particles into a ROS pose array message
    """
    def particle_to_pose(particle: np.array) -> Pose:
        return pose_to_pose_msg(Pose(*particle))

    return PoseArray(poses=[particle_to_pose(p) for p in particles])

def publish_image(map_png_file:str, cv_bridge:CvBridge, pub: rospy.Publisher) -> None:
    """
    Load the given image file and publish using the given publisher
    """
    img = cv2.imread(map_png_file)
    img_msg = cv_bridge.cv2_to_imgmsg(img, 'bgr8')
    pub.publish(img_msg)
    

    
class CarpetLocaliser():
    """
    Interface between incoming ROS messages (odom and camera image) and
    carpet based particle filter.
    """
    def __init__(self,
                 map_png_file: str,
                 map_cell_size: float,
                 classifier_param_file: str,
                 log_inputs=False):

        self.update_distance_threshold = 0.2  # m
        self.update_rotation_threshold = 0.1  # rad

        self.color_classifier = CarpetColorClassifier(classifier_param_file)
        self.log_inputs = log_inputs

        carpet = load_map_from_png(map_png_file, map_cell_size)

        self.particle_filter = CarpetBasedParticleFilter(carpet, log_inputs)
        self.cv_bridge = CvBridge()
        self.previous_odom = None

        self.pose_pub = rospy.Publisher("current_pose",
                                        Odometry,
                                        queue_size=10)
        self.particle_pub = rospy.Publisher(
            "particlecloud",
            PoseArray,
            queue_size=10,
        )

        # publish the carpet map on a latched image topic
        self.map_pub = rospy.Publisher("carpet_map", 
                                       Image,
                                       latch=True,
                                       queue_size=10)
        publish_image(map_png_file, self.cv_bridge, self.map_pub)

    def __del__(self):
        if self.log_inputs:
            log_path = "/tmp/localiser_input_log.pickle"
            self.particle_filter.write_input_log(log_path)
            rospy.loginfo(f"Saved localiser input log to {log_path}")

    def localisation_update(self,
                            odom_msg: Odometry,
                            img_msg: Image,
                            ground_truth_state: Optional[Odometry] = None):
        """
        Update localisation based on given odom,image pair
        Can optionally provide ground truth for logging purposes
        """

        # determine odom delta since last update
        current_odom = odom_msg_to_measurement(odom_msg)
        odom_delta = compute_odom_delta(current_odom, self.previous_odom)

        # perform particle filter update, only if robot has travelled beyond
        # a certain distance since the previous update
        distance_since_last_update = np.sqrt(odom_delta.dx**2 +
                                             odom_delta.dy**2)
        rotation_since_last_update = np.abs(odom_delta.dheading)

        if distance_since_last_update > self.update_distance_threshold or \
           rotation_since_last_update > self.update_rotation_threshold or \
           self.previous_odom is None:

            rospy.loginfo(f"odom: {odom_delta}")

            # determine detected color
            color_measurement, color_name = self._classify_image_color(img_msg)
            rospy.loginfo(f"detected color: {color_name}")

            # get ground truth pose if provided
            if ground_truth_state:
                ground_truth_pose = odom_msg_to_pose(ground_truth_state)
                rospy.loginfo(f"ground_truth: {ground_truth_pose}")
            else:
                ground_truth_pose = None

            # perform update
            self.particle_filter.update(odom_delta, color_measurement,
                                        ground_truth_pose)

            # create and publish odom msg representing current pose
            current_pose = Odometry()
            current_pose.pose.pose = pose_to_pose_msg(
                self.particle_filter.get_current_pose())
            current_pose.header.frame_id = "map"
            current_pose.header.stamp = img_msg.header.stamp
            current_pose.twist = odom_msg.twist
            self.pose_pub.publish(current_pose)

            # also publish current particles
            pose_array = particles_to_pose_array(
                self.particle_filter.get_particles())
            pose_array.header = current_pose.header
            self.particle_pub.publish(pose_array)

            self.previous_odom = current_odom

    def _classify_image_color(self,
                              img_msg: Image) -> Tuple[ColorMeasurement, str]:
        """
        invoke classifier on given image, returning color index and name (string)
        """
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg,
                                                desired_encoding='bgr8')
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

    # param 'subscribe_ground_truth': set true to subscribe to ground truth pose
    # (expected use-case = use with 'log_inputs' to log gazebo ground truth pose, for testing)
    subscribe_ground_truth = rospy.get_param("~subscribe_ground_truth",
                                             default=False)

    localiser = CarpetLocaliser(map_png_file, map_cell_size,
                                classifier_param_file, log_inputs)

    odom_sub = message_filters.Subscriber('odom', Odometry)
    image_sub = message_filters.Subscriber('image', Image)
    subs = [odom_sub, image_sub]

    if subscribe_ground_truth:
        ground_truth_sub = message_filters.Subscriber('ground_truth/state',
                                                      Odometry)
        subs.append(ground_truth_sub)

    time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        subs, 10, 0.2)
    time_synchronizer.registerCallback(localiser.localisation_update)

    def img_callback(img_msg: Image):
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
