#!/usr/bin/env python

import math
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import String
import rospy
from constants import *

SCALING_FACTOR = float(rospy.get_param('/tiago_action_manager/scaling_factor'))


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
    
    if where == torso_movements.UP or where == torso_movements.DOWN:
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [J_TORSO_TARGETUP if where == torso_movements.UP else J_TORSO_TARGETDOWN]
        point.velocities = []
        point.accelerations = []
        point.effort = []
        point.time_from_start = rospy.Duration(J_TORSO_TARGETTIME) #secs
        torso_cmd.points = [point]
        
    elif where == torso_movements.CENTRE:
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [J_TORSO_TARGETUP/2]
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
        point.time_from_start = rospy.Duration(J_HEAD_TARGETTIME) #secs
        head_cmd.points = [point]
        
    elif where == head_movements.UP or where == head_movements.DOWN:  
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [J_HEAD_2_TARGETUP if where == head_movements.UP else J_HEAD_2_TARGETDOWN, 0.0]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_TARGETTIME) #secs
        head_cmd.points = [point]
        
    elif where == head_movements.CENTRE:
        # Joint trajectory point to reach
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_TARGETTIME) #secs
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

        self.scaling_factor = 0.0
        self.goal = None
        self.robot_goal_distance = float("inf")

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
        
        # Goal and Robot pose subscriber
        self.sub_goal = rospy.Subscriber('/move_base/current_goal', PoseStamped, self.cb_goal)
        self.sub_robot_pos = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, self.cb_robot_pos)
    
    
    def cb_goal(self, msg : PoseStamped):
        """
        Callback for the Subscriber "sub_goal"

        Args:
            msg (PoseStamped): Robot current goal [map frame]
        """
        self.goal = msg.pose.position
        
        
    def cb_robot_pos(self, msg : PoseWithCovarianceStamped):
        """
        Callback for the Subscriber "sub_robot_pos"

        Args:
            msg (PoseWithCovarianceStamped): Robot current pose [map frame]
        """
        self.robot_pose = msg.pose.pose.position
        if self.goal is not None:
            self.robot_goal_distance = math.sqrt((self.robot_pose.x - self.goal.x)**2 + (self.robot_pose.y - self.goal.y)**2)
        
        
    def cb_action_listener(self, key_action):
        """
        Callback of the Subscriber "sub_key_action".
        Read keyboard input from /key_action topic and publish msgs to perform the corresponding movement.
        
        Args:
            key_action (String): data field containing the key 
        """
        if key_action.data == 'a':
            self.pub_head_action.publish(create_head_msg(head_movements.LEFT))
            while abs(self.head_1_pos - J_HEAD_1_TARGETLEFT) > 0.05: rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.RIGHT))
            while abs(self.head_1_pos - J_HEAD_1_TARGETRIGHT) > 0.05: rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.CENTRE))
            
        elif key_action.data == 'b':
            self.pub_head_action.publish(create_head_msg(head_movements.UP))
            while abs(self.head_2_pos - J_HEAD_2_TARGETUP) > 0.05: rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.DOWN))
            while abs(self.head_2_pos - J_HEAD_2_TARGETDOWN) > 0.05: rospy.sleep(0.1)
            self.pub_head_action.publish(create_head_msg(head_movements.CENTRE))
            
        elif key_action.data == 'c':
            self.pub_torso_action.publish(create_torso_msg(torso_movements.UP))
            while abs(self.torso_pos - J_TORSO_TARGETUP) > 0.05: rospy.sleep(0.1)
            self.pub_torso_action.publish(create_torso_msg(torso_movements.DOWN))
            while abs(self.torso_pos - J_TORSO_TARGETDOWN) > 0.05: rospy.sleep(0.1)
            self.pub_torso_action.publish(create_torso_msg(torso_movements.CENTRE))
            
        elif key_action.data == 'd':
            self.scaling_factor = SCALING_FACTOR
            # rospy.loginfo("scaling_factor = " + str(self.scaling_factor))
            
        elif key_action.data == 'e':
            self.scaling_factor = -SCALING_FACTOR
            # if self.scaling_factor - 0.1 > -1: # check in order to not invert the velocity sign
            #     self.scaling_factor = self.scaling_factor - 0.1
            # rospy.loginfo("scaling_factor = " + str(self.scaling_factor))

        
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
        if self.robot_goal_distance < DIST_THRES: self.scaling_factor = 0
        
        # scale the velocity
        new_msg = Twist()
        new_msg.linear.x = msg.linear.x + msg.linear.x * self.scaling_factor
        new_msg.linear.y = msg.linear.y + msg.linear.y * self.scaling_factor
        new_msg.linear.z = msg.linear.z + msg.linear.z * self.scaling_factor
        new_msg.angular.x = msg.angular.x + msg.angular.x * self.scaling_factor
        new_msg.angular.y = msg.angular.y + msg.angular.y * self.scaling_factor
        new_msg.angular.z = msg.angular.z + msg.angular.z * self.scaling_factor
        
        # publish new vel
        self.pub_cmd_vel.publish(new_msg)
    

if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    action_controller = ActionController()
        
    rospy.spin()