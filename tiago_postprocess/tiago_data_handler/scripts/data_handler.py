#!/usr/bin/env python

from enum import Enum
import math
import os
import random

import numpy as np
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import message_filters
import rospy
import pandas as pd
from shapely.geometry import *
from people_msgs.msg import People
import tf


FILENAME = str(rospy.get_param("/tiago_data_handler/bagname"))
DATAPATH = str(rospy.get_param("/tiago_data_handler/datapath"))
NODE_NAME = 'tiago_data_handler'
NODE_RATE = 100 #Hz


class Goal(Enum):
    X = (2.575, -1.604)
    Y = (2.575, 5.000)
    
GOAL = None


def get_2DPose(p: PoseWithCovarianceStamped):
    """
    Extracts x, y and theta from pose

    Args:
        p (PoseWithCovarianceStamped): pose

    Returns:
        tuple: x, y, theta
    """
    x = p.pose.pose.position.x
    y = p.pose.pose.position.y
    
    q = (
        p.pose.pose.orientation.x,
        p.pose.pose.orientation.y,
        p.pose.pose.orientation.z,
        p.pose.pose.orientation.w
    )
    
    m = tf.transformations.quaternion_matrix(q)
    _, _, yaw = tf.transformations.euler_from_matrix(m)
    return x, y, yaw
            

class DataHandler():
    """
    Class handling data
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.df_robot = pd.DataFrame(columns=['g_x', 'g_y', 'r_x', 'r_y', 'r_theta', 'r_v', 'r_omega'])
        self.people_dict = dict()      
        
        # Base subscriber
        self.sub_odom = message_filters.Subscriber('/mobile_base_controller/odom', Odometry)
                
        # Robot pose subscriber
        self.sub_robot_pos = message_filters.Subscriber('/robot_pose', PoseWithCovarianceStamped)
                
        # People subscriber
        self.sub_person_pos = message_filters.Subscriber('/people_tracker/people', People)
        
        # Init synchronizer and assigning a callback 
        self.ats = message_filters.ApproximateTimeSynchronizer([self.sub_odom, 
                                                                self.sub_robot_pos, 
                                                                self.sub_person_pos], 
                                                                queue_size = 100, slop = 1,
                                                                allow_headerless = True)

        self.ats.registerCallback(self.cb_handle_data)
        
        
    def people_handler(self, people: list, index: int):
        """
        Extracts people state (x, y, vel)

        Args:
            people (list): list of people
            index (int): current time step
        """
        for p in people:
            # Velocity
            vel = math.sqrt(p.velocity.x**2 + p.velocity.y**2)
            
            # Orientation
            theta = math.atan2(p.velocity.y, p.velocity.x)
            
            # Position
            if vel >= 0.75 and vel <= 2.00:
                if p.name not in self.people_dict:
                    self.people_dict[p.name] = pd.DataFrame(columns=["h_x", "h_y", "h_v", "h_theta"])
                self.people_dict[p.name].loc[index] = {"h_x": p.position.x,
                                                       "h_y": p.position.y,
                                                       "h_v": vel,
                                                       "h_theta": theta}
                
                
    def cb_handle_data(self, robot_odom: Odometry, 
                             robot_pose: PoseWithCovarianceStamped,
                             people: People):
        """
        Synchronized callback

        Args:
            robot_odom (Odometry): robot odometry
            robot_pose (PoseWithCovarianceStamped): robot pose
            people (People): tracked people
        """

        timestep = robot_odom.header.stamp
        
        # Robot 2D pose (x, y, theta)
        r_x, r_y, r_theta = get_2DPose(robot_pose)
                
        # Base linear & angular velocity
        base_vel = robot_odom.twist.twist.linear.x
        base_ang_vel = robot_odom.twist.twist.angular.z
              
        # appending new data row in robot Dataframe
        self.df_robot.loc[len(self.df_robot)] = {'time': timestep,
                                                 'g_x': GOAL[0], 'g_y': GOAL[1],
                                                 'r_x': r_x, 'r_y': r_y, 'r_theta': r_theta,
                                                 'r_v': base_vel, 'r_omega': base_ang_vel,
                                                 }
        
        self.people_handler(people.people, len(self.df_robot))      
        
            

if __name__ == '__main__':
    os.makedirs(DATAPATH, exist_ok=True)
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)

    # Goal definition
    GOAL = Goal.Y.value if int(FILENAME[-1]) % 2 == 0 else Goal.X.value
    
    data_handler = DataHandler()
        
    rospy.spin()
    if data_handler.people_dict:
        # FIXME: this take the tracked person with longest time-series. 
        # It should be changed to take the correct person. 
        len_tmp = {}
        for p in data_handler.people_dict:
            len_tmp[p] = len(data_handler.people_dict[p])
        p = max(len_tmp, key=len_tmp.get)
        rospy.logwarn("PERSON ID: %s"%p)
        ###############################################################
        

        df_complete = pd.concat([data_handler.df_robot, data_handler.people_dict[p]], axis = 1)
        df_complete.to_csv(DATAPATH + "/" + FILENAME + "_raw.csv")