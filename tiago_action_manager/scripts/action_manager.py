#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import rospy
from constants import *
import head_action as HA
import torso_action as TA



class ActionController():
    """
    Class handling different pre-defined movements of the TIAGo robot
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        # Head subscribers & publishers
        self.pub_head_action = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size = 10)
        self.sub_head_state = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, self.cb_head_state)
        
        # Torso subscribers & publishers
        self.pub_torso_action = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size = 10)
        self.sub_torso_state = rospy.Subscriber('/torso_controller/state', JointTrajectoryControllerState, self.cb_torso_state)
        
        # Base subscriber & publisher
        self.pub_cmd_vel = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 10)
        self.sub_tmp_cmd_vel = rospy.Subscriber('/mobile_base_controller/tmp_cmd_vel', Twist, self.cb_vel)
        
        # Action subscriber
        self.sub_key_action = rospy.Subscriber("/key_action", String, self.cb_action_listener)
               
        
    def cb_action_listener(self, key_action):
        """
        Callback of the Subscriber "sub_key_action".
        Read keyboard input from /key_action topic and publish msgs to perform the corresponding movement.
        
        Args:
            key_action (String): data field containing the key 
        """
        if key_action.data == 'a':
            self.pub_head_action.publish(HA.create_head_msg(head_movements.LEFT))
            while abs(self.head_1_pos - J_HEAD_1_TARGETLEFT) > 0.05: 
                rospy.sleep(0.1)
            self.pub_head_action.publish(HA.create_head_msg(head_movements.RIGHT))
            
        elif key_action.data == 'b':
            self.pub_head_action.publish(HA.create_head_msg(head_movements.UP))
            while abs(self.head_2_pos - J_HEAD_2_TARGETUP) > 0.05: 
                rospy.sleep(0.1)
            self.pub_head_action.publish(HA.create_head_msg(head_movements.DOWN))
            
        elif key_action.data == 'c':
            self.pub_torso_action.publish(TA.create_torso_msg(torso_movements.UP))
            while abs(self.torso_pos - J_TORSO_TARGETUP) > 0.05: 
                rospy.sleep(0.1)
            self.pub_torso_action.publish(TA.create_torso_msg(torso_movements.DOWN))
            
        elif key_action.data == '+':
            SCALING_FACTOR = SCALING_FACTOR + 0.1
            print("Velocity scaling factor:", SCALING_FACTOR)
        
        elif key_action.data == '-':
            SCALING_FACTOR = SCALING_FACTOR - 0.1
            print("Velocity scaling factor:", SCALING_FACTOR)
        
        
    def cb_head_state(self, msg):
        """
        Callback of the Subscriber "sub_head_state".
        Read head joints state from /head_controller/state topic
        """ 
        self.head_1_pos = msg.actual.positions[0]
        self.head_2_pos = msg.actual.positions[1]
        
        
    def cb_torso_state(self, msg):
        """
        Callback of the Subscriber "sub_torso_state".
        Read torso joint state from /torso_controller/state topic
        """ 
        self.torso_pos = msg.actual.positions[0]

    
    def cb_vel(self, msg : Twist):
        """
        Callback of the Subscriber "sub_tmp_cmd_vel".
        Scales the velocity read from /tmp_cmd_vel and publish the new velocity to /mobile_base_controller/cmd_vel
        
        Args:
            msg (Twist): Twist msg from topic /mobile_base_controller/tmp_cmd_vel
        """
        
        # scale the velocity
        new_msg = Twist()
        new_msg.linear.x = msg.linear.x - msg.linear.x * SCALAR_FACTOR
        new_msg.linear.y = msg.linear.y - msg.linear.y * SCALAR_FACTOR
        new_msg.linear.z = msg.linear.z - msg.linear.z * SCALAR_FACTOR
        new_msg.angular.x = msg.angular.x - msg.angular.x * SCALAR_FACTOR
        new_msg.angular.y = msg.angular.y - msg.angular.y * SCALAR_FACTOR
        new_msg.angular.z = msg.angular.z - msg.angular.z * SCALAR_FACTOR
        
        # publish new vel
        self.pub_cmd_vel.publish(new_msg)
    

if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    action_controller = ActionController()
        
    rospy.spin()