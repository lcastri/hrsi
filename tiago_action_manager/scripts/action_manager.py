#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import rospy
from constants import *


def create_torso_msg(where: torso_movements) -> JointTrajectory:
    """
    Creates torso msg for two pre-defined movements

    Args:
        where (torso_movements): UP/DOWN

    Returns:
        JointTrajectory: torso msg
    """
    torso_cmd = JointTrajectory()
       
    # Joint name to move
    torso_cmd.joint_names = [J_TORSO]
    
    # Joint trajectory point to reach
    point = JointTrajectoryPoint()
    point.positions = [J_TORSO_TARGETUP if where == torso_movements.UP else J_TORSO_TARGETDOWN]
    point.velocities = []
    point.accelerations = []
    point.effort = []
    point.time_from_start = rospy.Duration(J_TORSO_TARGETTIME) #secs
    torso_cmd.points = [point]
    
    return torso_cmd


def create_head_msg(where: head_movements) -> JointTrajectory:
    """
    Creates head msg for two pre-defined movements

    Args:
        where (head_movements): LEFT/RIGHT/UP/DOWN

    Returns:
        JointTrajectory: head msg
    """
    head_cmd = JointTrajectory()
    
    # REMINDER: 
    # J_HEAD_1 left & right
    # J_HEAD_2 up & down
    
    # Joint name to move
    head_cmd.joint_names = [J_HEAD_2, J_HEAD_1]
    
    if where == head_movements.LEFT or where == head_movements.RIGHT:
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [0.0, J_HEAD_1_TARGETLEFT if where == head_movements.LEFT else J_HEAD_1_TARGETRIGHT]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_1_TARGETTIME) #secs
        head_cmd.points = [point]
        
    elif where == head_movements.UP or where == head_movements.DOWN:  
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [J_HEAD_2_TARGETUP if where == head_movements.UP else J_HEAD_2_TARGETDOWN, 0.0]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_2_TARGETTIME) #secs
        head_cmd.points = [point]
    
    return head_cmd



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
            self.pub_head_action.publish(create_head_msg(head_movements.LEFT))
            while abs(self.head_1_pos - J_HEAD_1_TARGETLEFT) > 0.05: 
                rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.RIGHT))
            
        elif key_action.data == 'b':
            self.pub_head_action.publish(create_head_msg(head_movements.UP))
            while abs(self.head_2_pos - J_HEAD_2_TARGETUP) > 0.05: 
                rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.DOWN))
            
        elif key_action.data == 'c':
            self.pub_torso_action.publish(create_torso_msg(torso_movements.UP))
            while abs(self.torso_pos - J_TORSO_TARGETUP) > 0.05: 
                rospy.sleep(0.1)
            self.pub_torso_action.publish(create_torso_msg(torso_movements.DOWN))
            
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