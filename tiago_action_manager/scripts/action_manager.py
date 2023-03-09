#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from control_msgs.msg import JointTrajectoryControllerState
import utils
import rospy
from constants import *
import head_action as HA
import torso_action as TA



class Head_controller():
    def __init__(self):
        # Head subscribers & publishers
        self.pub_head_action = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size = 10)
        self.sub_head_state = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, self.cb_head_state)
        
        # Torso subscribers & publishers
        self.pub_torso_action = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size = 10)
        
        # Base subscribers & publishers
        self.pub_base_action = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 10)
        
        # Action subscriber
        self.sub_key_action = rospy.Subscriber("/key_action", String, self.cb_action_listener)
        
        
    def cb_action_listener(self, key_action):
    # global action_requested
    
        if key_action.data == 'a':
            print("Head left and right")
            # head_LR()
            # action_requested = True
            self.sub_key_action.unregister()
            self.pub_head_action.publish(HA.create_head_msg(head_movements.LEFT))
            self.sub_key_action = rospy.Subscriber("/key_action", String, self.cb_action_listener)

            print("cacca")
            
        elif key_action.data == 'b':
            print("Head up and down")
            # head_UD()
        elif key_action.data == 'c':
            print("Torso up and down")
            # torso_UD()
        
        
    def cb_head_state(self, msg):  
        # global action_requested 
        print("Main: reading head pos")
        # HEAD_1_POS = data.actual.positions[0]
        # HEAD_2_POS = data.actual.positions[1]
        # if abs(HEAD_1_POS - J_HEAD_1_LEFT) < 0.05: 
        # # if action_requested and abs(HEAD_1_POS - J_HEAD_1_LEFT) < 0.05: 
        #     print("culo")

        #     self.pub_head_action.publish(HA.create_head_msg(head_movements.RIGHT))
        #     # action_requested = False


# action_requested = False

# def head_LR():
#     print("Thread head_LR starting")
#     """
#     moves TIAGo's head left - wait 1 sec - moves TIAGo's head left
#     """
#     # global HEAD_1_POS
#     pub_head_action.publish(HA.create_head_msg(head_movements.LEFT))
#     if pos_reached: 
#     # print("Thread head_LR left action sent")

#     while abs(HEAD_1_POS - J_HEAD_1_LEFT) > 0.05: 
#         print("Thread head_LR sleeping")
#         print(HEAD_1_POS - J_HEAD_1_LEFT)
#         time.sleep(0.1)
#     rospy.sleep(1)
#     pub_head_action.publish(HA.create_head_msg(head_movements.RIGHT))
#     while abs(HEAD_1_POS - J_HEAD_1_RIGHT) > 0.05: rospy.sleep(0.1)


# def head_UD():
#     """
#     moves TIAGo's head up - wait 1 sec - moves TIAGo's head down
#     """
#     global HEAD_2_POS
#     pub_head_action.publish(HA.create_head_msg(head_movements.UP))
#     rospy.sleep(1)
#     pub_head_action.publish(HA.create_head_msg(head_movements.DOWN))
    
    
# def torso_UD():
#     """
#     moves TIAGo's torso up - wait 1 sec - moves TIAGo's torso down
#     """
#     pub_torso_action.publish(TA.create_torso_msg(torso_movements.UP))
#     rospy.sleep(1)
#     pub_torso_action.publish(TA.create_torso_msg(torso_movements.DOWN))
    
    
# def publish_base_command():
#     base_cmd = Twist()
    

if __name__ == '__main__':    
    try:
        # Print node info
        utils.print_node_info()
        
        # Init node
        rospy.init_node(NODE_NAME)
        
        # Set node rate
        rate = rospy.Rate(NODE_RATE)
        
        head_c = Head_controller()
        
        while not rospy.is_shutdown():
            print("CIAO")
            rate.sleep()

        # # Head subscribers & publishers
        # pub_head_action = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size = 10)
        # rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, head_state_cb)
        
        # # Torso subscribers & publishers
        # pub_torso_action = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size = 10)
        
        # # Base subscribers & publishers
        # pub_base_action = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 10)
        
        # # Action subscriber
        # rospy.Subscriber("/key_action", String, action_listener)
        
        # rospy.spin()
        
    except KeyboardInterrupt:
        pass