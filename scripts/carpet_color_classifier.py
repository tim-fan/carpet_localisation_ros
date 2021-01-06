#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from carpet_color_classification import CarpetColorClassifier

    
def run_classifier():

    rospy.init_node('carpet_color_classifier')
    rospy.loginfo("initialised")
    rospy.loginfo("deprecation warning! This will be merged into the localisation package")

    classifier_param_file = rospy.get_param("~classifier_param_file")
    classifier = CarpetColorClassifier(classifier_param_file)

    cv_bridge = CvBridge()


    def img_callback(img_msg:Image):
        cv_image = cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        color_index, color_name = classifier.classify(cv_image)
        rospy.loginfo(f"detected color: {color_name}")
        

    rospy.Subscriber("image", Image, img_callback)

    rospy.spin()

if __name__ == '__main__':
    run_classifier()

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