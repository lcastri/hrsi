#!/usr/bin/env python

import os
import rospy
from std_msgs.msg import Bool

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

        # Joy subscriber
        self.sub_joy_priority = rospy.Subscriber("/joy_priority", Bool, self.cb_prioritytrigger)
        
    
    def cb_prioritytrigger(self, msg : Bool):

        if msg.data:
            os.system("pal-stop " + PAL_LED_MANAGER)
            rospy.sleep(1)
            os.system("rosservice call /mm11/led/set_strip_animation 0 2 100 5 250 0 0 0 0 255")
            os.system("rosservice call /mm11/led/set_strip_animation 1 2 100 5 250 0 0 0 0 255")
            # os.system("rosservice call /mm11/led/set_strip_flash 0 500 100 0 0 0 255 255 255")
            # os.system("rosservice call /mm11/led/set_strip_flash 1 500 100 0 0 0 255 255 255")
        else:
            os.system("pal-start " + PAL_LED_MANAGER)

    
if __name__ == '__main__':
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    action_controller = JoyPriorityLed()
        
    rospy.spin()