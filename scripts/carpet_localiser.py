#!/usr/bin/env python
from typing import Tuple, Optional
import pickle
import numpy as np
import cv2
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Pose as PoseMsg
from nav_msgs.msg import Odometry, OccupancyGrid
from cv_bridge import CvBridge
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from carpet_color_classification import CarpetColorClassifier
from cbl_particle_filter.filter import CarpetBasedParticleFilter, Pose, OdomMeasurement, add_poses
import cbl_particle_filter.colors as colors
from cbl_particle_filter.carpet_map import load_map_from_png, CarpetMap


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
        return pose_to_pose_msg(Pose(*particle[0:3]))

    return PoseArray(poses=[particle_to_pose(p) for p in particles])


def publish_image(map_png_file: str, cv_bridge: CvBridge,
                  pub: rospy.Publisher) -> None:
    """
    Load the given image file and publish using the given publisher
    """
    img = cv2.imread(map_png_file)
    img_msg = cv_bridge.cv2_to_imgmsg(img, 'bgr8')
    pub.publish(img_msg)


def write_color_name_on_cv_img(cv_img: np.ndarray,
                               color: colors.Color) -> np.ndarray:
    """
    Write the name of the given color onto the image
    """
    # define text colors
    color_index_to_bgr_tuple = {
        colors.BLACK.index: (0, 0, 0),
        colors.LIGHT_BLUE.index: (255, 204, 51),
        colors.BEIGE.index: (169, 214, 213),
        colors.DARK_BLUE.index: (204, 51, 0),
        colors.UNCLASSIFIED.index: (100, 100, 100)
    }
    cv2.rectangle(cv_img, (10, 220), (310, 150), (255, 255, 255),
                  -1)  #background
    return cv2.putText(img=cv_img,
                       text=color.name,
                       org=(10, 200),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.6,
                       color=color_index_to_bgr_tuple[color.index],
                       thickness=10)


def publish_carpet_map_outline(carpet_map: CarpetMap, pub: rospy.Publisher):
    """
    Publish an occupancy grid from the given carpet map.
    The occupancy grid will show clear space where there is carpet, otherwise
    occupied space
    """
    o_grid = OccupancyGrid()
    o_grid.header.frame_id = "map"
    color_indices = [color.index for color in colors.COLORS]
    o_grid.data = [
        0 if elem in color_indices else 100
        for elem in np.flipud(carpet_map.grid).flatten()
    ]
    height, width = carpet_map.grid.shape
    o_grid.info.height = height
    o_grid.info.width = width
    o_grid.info.resolution = carpet_map.cell_size
    o_grid.info.origin.orientation.w = 1

    pub.publish(o_grid)


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

        resample_proportion = 0

        self.color_classifier = CarpetColorClassifier(classifier_param_file)
        self.log_inputs = log_inputs

        carpet = load_map_from_png(map_png_file, map_cell_size)

        self.particle_filter = CarpetBasedParticleFilter(carpet, log_inputs, resample_proportion=resample_proportion)
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

        self.classified_image_pub = rospy.Publisher("classified_image",
                                                    Image,
                                                    queue_size=10)

        # publish the carpet map on a latched image topic
        self.map_pub = rospy.Publisher(
            "carpet_map",
            Image,
            latch=True,
            queue_size=10,
        )
        publish_image(map_png_file, self.cv_bridge, self.map_pub)

        self.occupancy_pub = rospy.Publisher(
            "carpet_map_outline",
            OccupancyGrid,
            latch=True,
            queue_size=10,
        )
        publish_carpet_map_outline(carpet, self.occupancy_pub)

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

        # prepare output pose message (odom)
        current_pose = Odometry()
        current_pose.header.frame_id = "map"
        current_pose.header.stamp = img_msg.header.stamp
        current_pose.twist = odom_msg.twist

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
            color = self._classify_image_color(img_msg)
            rospy.loginfo(f"detected color: {color.name}")

            # get ground truth pose if provided
            if ground_truth_state:
                ground_truth_pose = odom_msg_to_pose(ground_truth_state)
                rospy.loginfo(f"ground_truth: {ground_truth_pose}")
            else:
                ground_truth_pose = None

            # perform update
            self.particle_filter.update(odom_delta, color, ground_truth_pose)

            # get current pose from the particle filter
            current_pose.pose.pose = pose_to_pose_msg(
                self.particle_filter.get_current_pose())

            # also publish current particles
            self._publish_particles(header=current_pose.header)

            self.previous_odom = current_odom

        else:
            # set current pose as particle filter pose at last update
            # plus accumulated odom since then
            pf_pose = self.particle_filter.get_current_pose()

            updated_pose_array = add_poses(
                current_poses=np.array([[
                    pf_pose.x,
                    pf_pose.y,
                    pf_pose.heading,
                ]]),
                pose_increments=np.array([[
                    odom_delta.dx,
                    odom_delta.dy,
                    odom_delta.dheading,
                ]])
            )[0] # yapf: disable

            updated_pose = Pose(x=updated_pose_array[0],
                                y=updated_pose_array[1],
                                heading=updated_pose_array[2])
            current_pose.pose.pose = pose_to_pose_msg(updated_pose)

        # publish current location
        self.pose_pub.publish(current_pose)

    def seed(self, pose_msg:PoseWithCovarianceStamped) -> None:
        seed_pose = Pose(
            x=pose_msg.pose.pose.position.x,
            y=pose_msg.pose.pose.position.y,
            heading=yaw_from_quaternion_msg(pose_msg.pose.pose.orientation),
        )
        self.particle_filter.seed(seed_pose)
        self.previous_odom = None
        self._publish_particles(header=pose_msg.header)

    def _publish_particles(self, header:Header) -> None:
            pose_array = particles_to_pose_array(
                self.particle_filter.get_particles())
            pose_array.header = header
            self.particle_pub.publish(pose_array)

    def _classify_image_color(self, img_msg: Image) -> colors.Color:
        """
        invoke classifier on given image, returning color 
        """
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg,
                                                desired_encoding='bgr8')
        _, color_name = self.color_classifier.classify(cv_image)
        color = colors.color_from_name[color_name]

        # republish the image with the classification written on it, for visualisation/debug
        cv_image = write_color_name_on_cv_img(cv_image, color)
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.classified_image_pub.publish(img_msg)

        return color


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

    init_pose_sub = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, localiser.seed )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("INTERRUPT!")

    del localiser
    print("DONE!")


if __name__ == '__main__':
    run_localisation()
