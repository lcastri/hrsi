#!/usr/bin/env python

import math
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import rospy
from enum import Enum


class action_strategy(Enum):
    RANDOM = 0
    KEYBOARD = 1
    
class head_movements(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CENTRE = 4
    
class torso_movements(Enum):
    UP = 0
    DOWN = 1
    CENTRE = 2

NODE_NAME = 'action_manager'
NODE_RATE = 10 #Hz

J_HEAD_1 = "head_1_joint"
J_HEAD_2 = "head_2_joint"
J_HEAD_1_TARGETLEFT = 1.2
J_HEAD_1_TARGETRIGHT = -1.2
J_HEAD_TARGETTIME = 1.5
J_HEAD_2_TARGETUP = 0.7
J_HEAD_2_TARGETDOWN = -1

J_TORSO = "torso_lift_joint"
J_TORSO_TARGETUP = 0.35
J_TORSO_TARGETDOWN = 0.025
J_TORSO_TARGETTIME = 1.5

DIST_THRES = 2

SCALING_FACTOR = float(rospy.get_param('/tiago_action_manager/scaling_factor'))

POINT_X = PoseStamped()
POINT_X.header.frame_id = "map"
POINT_X.pose.position.x = 2.575
POINT_X.pose.position.y = -1.604
POINT_X.pose.position.z = 0.0
POINT_X.pose.orientation.x = 0.0
POINT_X.pose.orientation.y = 0.0
POINT_X.pose.orientation.z = 0.707
POINT_X.pose.orientation.w = 0.707
            
POINT_Y = PoseStamped()
POINT_Y.header.frame_id = "map"
POINT_Y.pose.position.x = 2.575
POINT_Y.pose.position.y = 5.000
POINT_Y.pose.position.z = 0.0
POINT_Y.pose.orientation.x = 0.0
POINT_Y.pose.orientation.y = 0.0
POINT_Y.pose.orientation.z = -0.707
POINT_Y.pose.orientation.w = 0.707


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
        self.robot_pose = None

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
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size = 10)
        
        # Laser scan subscriber
        rospy.Subscriber('/scan', LaserScan, self.cb_closest_wall)


    def build_wall_goal(self):

        if self.goal.position.y > 0:
            wall_y = self.robot_pose.position.y + 0.75
        else:
            wall_y = self.robot_pose.position.y - 0.75

        closest_wall = PoseStamped()

        # Building the goal with the closest wall cordinates
        closest_wall.header.frame_id = "map"
        closest_wall.pose.position.x = self.wall_x
        closest_wall.pose.position.y = wall_y
        closest_wall.pose.position.z = 0.0
        closest_wall.pose.orientation.x = self.robot_pose.orientation.x
        closest_wall.pose.orientation.y = self.robot_pose.orientation.y
        closest_wall.pose.orientation.z = self.robot_pose.orientation.z
        closest_wall.pose.orientation.w = self.robot_pose.orientation.w
        return closest_wall

    
    
    def cb_closest_wall(self, msg : LaserScan):
        """
        _summary_

        Args:
            msg (LaserScan): _description_
        """

        if self.robot_pose is not None:
        
            # Find the minimum range and angle of the laser scan data
            min_range = min(msg.ranges)
            min_range_idx = msg.ranges.index(min_range)
            min_range_angle = msg.angle_min + min_range_idx * msg.angle_increment
            
            # Calculate the x,y coordinates of the closest wall in the map frame
            self.wall_x = self.robot_pose.position.x + min_range * math.cos(min_range_angle)
        

    def cb_goal(self, msg : PoseStamped):
        """
        Callback for the Subscriber "sub_goal"

        Args:
            msg (PoseStamped): Robot current goal [map frame]
        """
        self.goal = msg.pose
        
        
    def cb_robot_pos(self, msg : PoseWithCovarianceStamped):
        """
        Callback for the Subscriber "sub_robot_pos"

        Args:
            msg (PoseWithCovarianceStamped): Robot current pose [map frame]
        """
        self.robot_pose = msg.pose.pose
        if self.goal is not None:
            self.robot_goal_distance = math.sqrt((self.robot_pose.position.x - self.goal.position.x)**2 + (self.robot_pose.position.y - self.goal.position.y)**2)
        
        
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
            
        elif key_action.data == 'e':
            self.scaling_factor = -SCALING_FACTOR
            
        elif key_action.data == 'f':
            print("PREVIOUS GOAL")
            print(self.current_goal)

            print("CLOSEST WALL")
            closest_wall = self.build_wall_goal()
            print(closest_wall)

            self.pub_goal.publish(closest_wall)
            while self.robot_goal_distance > 0.1: rospy.sleep(0.1)
            print("CLOSEST WALL reached")
            self.pub_goal.publish(self.current_goal)

        elif key_action.data == 'x':
            self.current_goal = POINT_X
            self.pub_goal.publish(POINT_X)
            
        elif key_action.data == 'y':
            self.current_goal = POINT_Y
            self.pub_goal.publish(POINT_Y)

        
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