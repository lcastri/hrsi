#!/usr/bin/env python

import os
import rospy
from std_msgs.msg import Bool
from mm11_msgs.srv import SetStripAnimation, SetStripAnimationRequest
import rosnode


NODE_NAME = 'joy_priority_led'
NODE_RATE = 10 #Hz
PAL_LED_MANAGER = "pal_led_manager"


class JoyPriorityLed():
    """
    Class handling different pre-defined movements of the TIAGo robot
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        self.just_started = True

        # Joy subscriber
        self.sub_joy_priority = rospy.Subscriber("/joy_priority", Bool, self.cb_prioritytrigger)
        
    
    def cb_prioritytrigger(self, msg : Bool):
        if msg.data:
            os.system("pal-stop " + PAL_LED_MANAGER)
            srv = rospy.ServiceProxy("/mm11/led/set_strip_animation", SetStripAnimation)
            res = srv(SetStripAnimationRequest(port = 0, 
                                               animation_id = 2, 
                                               param_1 = 100, 
                                               param_2 = 5, 
                                               r_1 = 250, 
                                               g_1 = 0,
                                               b_1 = 0, 
                                               r_2 = 0, 
                                               g_2 = 0, 
                                               b_2 = 255))
            res = srv(SetStripAnimationRequest(port = 1, 
                                               animation_id = 2, 
                                               param_1 = 100, 
                                               param_2 = 5, 
                                               r_1 = 250, 
                                               g_1 = 0,
                                               b_1 = 0, 
                                               r_2 = 0, 
                                               g_2 = 0, 
                                               b_2 = 255))
        else:
            if not self.just_started:
                os.system("pal-start " + PAL_LED_MANAGER)
            else:
                self.just_started = False

    
if __name__ == '__main__':
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)

    while rosnode.rosnode_ping("pal_led_manager", 1) is False: rospy.sleep(0.1)

    joy_priority_led = JoyPriorityLed()
    
    rospy.spin()
