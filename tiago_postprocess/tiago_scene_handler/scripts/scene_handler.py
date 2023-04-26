#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os


VIDEOPATH = '/'.join([str(rospy.get_param("/tiago_scene_handler/scenepath")), 
                      str(rospy.get_param("/tiago_scene_handler/bagname")),
                      "scene.mp4"
                     ])
IMAGEPATH = '/'.join([str(rospy.get_param("/tiago_scene_handler/scenepath")), 
                      str(rospy.get_param("/tiago_scene_handler/bagname")),
                      "images"
                     ])
NODE_NAME = 'tiago_scene_handler'
NODE_RATE = 30 #Hz


class SceneHandler():
    def __init__(self):
        self.bridge = CvBridge()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Change the codec as per your requirement
        self.video_writer = cv2.VideoWriter(VIDEOPATH, fourcc, 30.0, (640, 480)) # Change the resolution as per your requirement
        self.count = 0
        
        # Image subscriber
        self.sub_person_pos = rospy.Subscriber("/camera/color/image_raw", Image, self.cb_image_handler, callback_args=0)
        
        
    def cb_image_handler(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.video_writer.write(cv_image)
        cv2.imwrite(os.path.join(IMAGEPATH, f"image{self.count:05}.jpg"), cv_image)
        self.count += 1


if __name__ == '__main__':
    try:
        os.makedirs(IMAGEPATH, exist_ok=True)
        rospy.init_node(NODE_NAME, anonymous=True)
        rate = rospy.Rate(NODE_RATE) # Change the rate as per your requirement
        scene_handler = SceneHandler()
        while not rospy.is_shutdown():
            rate.sleep()

        scene_handler.video_writer.release()
    except rospy.ROSInterruptException:
        pass

# Convert the video to MP4 using FFmpeg
os.system(f'ffmpeg -i {VIDEOPATH} -vcodec libx264 -crf 25 -y {VIDEOPATH}.mp4')