# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import sys

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# --image_topic /camera/color/image_raw or --image_topic =/camera/color/image_raw 
# --output_dir ../record/observations/color
# --bag_file_dir ../bag_real_images_VAE
# --bag_file scene_work_pc_1.bag


# run outside conda: python3 bag_read.py --bag_file sanitising_0.bag

def main():
    """Extract a folder of images from a rosbag.
    """
    level_of_help = 1

    bag_file_dir = '../bag_real_images_VAE/new_recordings/cleaning'
    output_dir = '../record/observations/color/cleaning'
    image_topic = '/camera/color/image_raw' # for some bags it will start with /camera/... for others as it starts with camera/...

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    #parser.add_argument("--bag_file_dir", help="Input ROS bag dir.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    #parser.add_argument("--output_dir", help="Output directory.")
    #parser.add_argument("--image_topic", help="Image topic.")

    args = parser.parse_args()

    #sys.path.append(args.output_dir)

    os.makedirs(os.path.join(output_dir, args.bag_file), exist_ok=True)

    print("Extract images from {} on topic {} into {}".format(args.bag_file, image_topic, output_dir))

    bag = rosbag.Bag(os.path.join(bag_file_dir, args.bag_file), "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        if (count % 10 == 0):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            cv2.imwrite(os.path.join(output_dir, args.bag_file, "frame{}-{}.png".format(count, level_of_help)), cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            print("Wrote image {}".format(count))

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()