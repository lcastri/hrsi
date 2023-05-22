#!/usr/bin/env python

import rospy
from people_msgs.msg import People
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose2D, PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal
import math
import tf
import numpy as np

NODE_NAME = 'person_following'
NODE_RATE = 100 # [Hz]
DES_DIST = 1.5 
# DES_DIST = str(rospy.get_param("/person_following/des_dist"))



class PersonFollowing():
    """
    Person Following class
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.personID_to_follow = None
        self.people = None
        self.robot_pose = Pose2D()
        
        # Robot pose subscriber
        self.sub_robot_pos = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, callback = self.cb_robot_pose)
        
        # Person to follow subscriber
        self.sub_person_to_follow = rospy.Subscriber('/person_to_follow', String, callback = self.cb_person_to_follow)
        
        # People subscriber
        self.sub_people = rospy.Subscriber('/people_tracker/people', People, callback = self.cb_people)
        
        # Robot goal publisher
        # self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size = 10)
        self.pub_goal = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
            
        
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
        
        
    def cb_person_to_follow(self, id: String):
        """
        Stores the person id to follow

        Args:
            id (String): Person ID detected
        """
        # check if old person ID still exist in the people list
        if self.personID_to_follow is None or not any(person.name == self.personID_to_follow for person in self.people):
            # If the old person ID is still detected, self.personID_to_follow not updated -> the robot will continue to follow the same person
            # If the old person ID is not detected anymore, self.personID_to_follow is updated -> the robot will follow the new person
            self.personID_to_follow = id.data

        rospy.logdebug("Person to follow ID" + self.personID_to_follow)

        
    def cb_people(self, data: People):
        """
        Stores people

        Args:
            data (People): people topic from people tracker
        """
        self.people = data.people
        
        
    def calculate_goal(self, person_pos):
        """
        Calculates goal pos and orientation

        Args:
            person_pos (list): [x, y]

        Returns:
            array, float: goal pos and orientation
        """
        person_pos = np.array(person_pos)
        robot_pos = np.array([self.robot_pose.x, self.robot_pose.y])
        
        # Calculate the vector from the robot to the person
        vector_to_person = person_pos - robot_pos
        # vector_to_person = (person_pos[0] - self.robot_pose.x, person_pos[1] - self.robot_pose.y)

        # Calculate the distance from the robot to the person
        distance_to_person = math.sqrt(vector_to_person[0]**2 + vector_to_person[1]**2)

        # Normalize the vector to the desired distance
        normalized_vector = vector_to_person / distance_to_person

        # Calculate the orientation needed to reach the person
        goal_orientation = math.atan2(normalized_vector[1], normalized_vector[0])

        # Calculate the goal position based on the desired distance
        goal_position = person_pos - DES_DIST * normalized_vector

        return goal_position, goal_orientation    
    
        
    def send_goal(self, goal_position, goal_orientation):
        """
        Creates goal msg

        Args:
            goal_position (array): x, y
            goal_orientation (float): theta
        """
        goal_msg = MoveBaseActionGoal()
        goal_msg.goal.target_pose.header.frame_id = "map"
        goal_msg.goal.target_pose.pose.position.x = goal_position[0]
        goal_msg.goal.target_pose.pose.position.y = goal_position[1]
        goal_msg.goal.target_pose.pose.orientation.z = math.sin(goal_orientation / 2)
        goal_msg.goal.target_pose.pose.orientation.w = math.cos(goal_orientation / 2)
        
        # goal_msg = PoseStamped()
        # goal_msg.header.frame_id = "map"
        # goal_msg.pose.position.x = goal_position[0]
        # goal_msg.pose.position.y = goal_position[1]
        # goal_msg.pose.orientation.z = math.sin(goal_orientation / 2)
        # goal_msg.pose.orientation.w = math.cos(goal_orientation / 2)
        
        # Publish the goal position and orientation to the navigation system
        self.pub_goal.publish(goal_msg)
        
    
    def follow_person(self):
        """
        Calculates and publishes the goal position and orientation when personID_to_follow is in people list
        """
        person_position = None
        
        if self.people is not None:
            for person in self.people:
                if person.name == self.personID_to_follow:
                    person_position = [person.position.x, person.position.y]
                    break
            
            if person_position is not None:
                # Calculate the goal position and orientation based on the desired distance
                goal_position, goal_orientation = self.calculate_goal(person_position)

                # Send the goal position and orientation to the navigation system
                self.send_goal(goal_position, goal_orientation)
            
        
if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    person_detector = PersonFollowing()
    
    while not rospy.is_shutdown():
        person_detector.follow_person()
        rate.sleep()