#!/usr/bin/env python

import rospy
from people_msgs.msg import People
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose2D
from std_msgs.msg import String

import tf
import math

NODE_NAME = 'person_detection'
NODE_RATE = 100 # [Hz]
FIELD_OF_VIEW = 60 # [°]
DIST_THRESH = 3 # [m]
# FIELD_OF_VIEW = str(rospy.get_param("/person_detection/field_of_view"))
# DIST_THRESH = str(rospy.get_param("/person_detection/dist_threshold"))


def wrapToPi(angle):
    """
    Adjust the angle to be within the range of -pi to pi

    Args:
        angle (float): Angle in rads

    Returns:
        float: Angle in rads mapped within the range of -pi to pi
    """
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


class PersonDetector():
    """
    Person Detector class
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.robot_pose = Pose2D()
        self.personID_to_follow = None
        
        # Robot pose subscriber
        self.sub_robot_pos = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, callback = self.cb_robot_pose)
        
        # People subscriber
        self.sub_person_pos = rospy.Subscriber('/people_tracker/people', People, callback = self.cb_people)
        
        self.pub_personID_to_follow = rospy.Publisher('/person_to_follow', String, queue_size=10)

        
        
    def cb_robot_pose(self, p: PoseWithCovarianceStamped):
        """
        from 3D to 2D robot pose

        Args:
            p (PoseWithCovarianceStamped): 3D robot pose
        """
        
        q = (
            p.pose.pose.orientation.x,
            p.pose.pose.orientation.y,
            p.pose.pose.orientation.z,
            p.pose.pose.orientation.w
        )
        
        m = tf.transformations.quaternion_matrix(q)
        
        self.robot_pose.x = p.pose.pose.position.x
        self.robot_pose.y = p.pose.pose.position.y
        self.robot_pose.theta = tf.transformations.euler_from_matrix(m)[2]

        
        
    def cb_people(self, data: People):
        """
        Handles the people topic from the tracker and identifies the person to follow

        Args:
            data (People): people topic from the tracker
        """
        if self.robot_pose.x is not None:
            closest_distance = float('inf')  # Initialize with a large value
            person_to_follow = None
            
            for person in data.people:
                # Calculate distance (assuming position is in meters)
                distance = math.dist([self.robot_pose.x, self.robot_pose.y], [person.position.x, person.position.y])
                
                # Calculate angle between person and robot
                angle = math.atan2(person.position.y - self.robot_pose.y, person.position.x - self.robot_pose.x)
                
                # Convert the angle to robot-centric coordinates
                relative_angle = wrapToPi(angle - self.robot_pose.theta)
                
                # Check if the person is within the desired field of view angle range
                if abs(relative_angle) <= math.radians(FIELD_OF_VIEW / 2) and distance <= DIST_THRESH:
                    if distance < closest_distance:
                        closest_distance = distance
                        person_to_follow = person
                    
            if person_to_follow is not None:
                self.personID_to_follow = person_to_follow.name
                rospy.logwarn("Person ID to follow: " + person_to_follow.name)
                
                self.pub_personID_to_follow.publish(self.personID_to_follow)
            else:
                rospy.logwarn("No person to follow identified")

    

if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    person_detector = PersonDetector()
        
    rospy.spin()