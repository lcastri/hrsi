#!/usr/bin/env python

import math
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
import message_filters
import rospy
import pandas as pd
import os


FILENAME = str(rospy.get_param("/tiago_data_handler/bagname"))
DATAPATH = str(rospy.get_param("/tiago_data_handler/datapath"))
NODE_NAME = 'tiago_data_handler'
NODE_RATE = 100 #Hz


class DataHandler():
    """
    Class handling data
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        self.df_pos = pd.DataFrame(columns = ['x_r', 'y_r', 'x_h', 'y_h'])
        self.df = pd.DataFrame(columns = ['vel_h1', 'vel_h2', 'vel_t', 'vel_r', 'omega_r','d_hr','v_h', 'theta_hr', 'risk'])
        
        # Head subscriber
        self.sub_head_state = message_filters.Subscriber("/head_controller/state", JointTrajectoryControllerState)
        
        # Torso subscriber
        self.sub_torso_state = message_filters.Subscriber('/torso_controller/state', JointTrajectoryControllerState)
        
        # Base subscriber
        self.sub_cmd_vel = message_filters.Subscriber('/mobile_base_controller/cmd_vel', Twist)
                
        # Robot pose subscriber
        self.sub_robot_pos = message_filters.Subscriber('/robot_pose', PoseWithCovarianceStamped)
        
        # Closest person pose subscriber
        self.sub_person_pos = message_filters.Subscriber('/people_tracker/pose', PoseStamped)
        
        # Init synchronizer and assigning a callback 
        self.ats = message_filters.ApproximateTimeSynchronizer([self.sub_head_state, 
                                                                self.sub_torso_state, 
                                                                self.sub_cmd_vel, 
                                                                self.sub_robot_pos, 
                                                                self.sub_person_pos], 
                                                                queue_size=100, slop=1)
        self.ats.registerCallback(self.cb_handle_data)
    
    
    def cb_handle_data(self, head_state: JointTrajectoryControllerState, 
                             torso_state: JointTrajectoryControllerState, 
                             robot_vel: Twist, 
                             robot_pose: PoseWithCovarianceStamped,
                             person_pose: PoseStamped):
        
        robot_pos_x = robot_pose.pose.pose.position.x
        robot_pos_y = robot_pose.pose.pose.position.y
        person_pos_x = person_pose.pose.position.x
        person_pos_y = person_pose.pose.position.y
        self.df_pos.loc[len(self.df_pos)] = {'x_r':robot_pos_x, 
                                             'y_r':robot_pos_y, 
                                             'x_h':person_pos_x, 
                                             'y_h':person_pos_y}
        
        head1_vel = head_state.actual.velocities[0]
        head2_vel = head_state.actual.velocities[1]
        torso_vel = torso_state.actual.velocities[0]
        base_vel = robot_vel.linear.x
        base_ang_vel = robot_vel.angular.z
        d_hr = math.dist([robot_pos_x, robot_pos_y], [person_pos_x, person_pos_y])
        # #TODO: compute v_h
        # #TODO: compute risk
        # #TODO: compute angle 
        self.df.loc[len(self.df)] = {'vel_h1':head1_vel, 
                                     'vel_h2':head2_vel, 
                                     'vel_t':torso_vel, 
                                     'vel_r':base_vel, 
                                     'omega_r':base_ang_vel,
                                     'd_hr': d_hr}
        
            

if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    data_handler = DataHandler()
        
    rospy.spin()
        
    data_handler.df_pos.to_csv( DATAPATH + "/" + FILENAME + "_XY.csv")
    data_handler.df.to_csv( DATAPATH + "/" + FILENAME + ".csv")