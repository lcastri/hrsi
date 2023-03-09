#!/usr/bin/env python
from __future__ import print_function
from constants import *
import rospy
from std_msgs.msg import String
import utils

import sys
from select import select

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


def getKey(settings, timeout):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__=="__main__":
    settings = saveTerminalSettings()

    # Print node info
    utils.print_node_info()
    
    # Init node
    rospy.init_node(NODE_NAME, anonymous = True)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    # Print key instructions
    utils.print_instruction()
    
    # Publisher definition
    pub_key_action = rospy.Publisher(PUB_TOPIC_NAME, String, queue_size = 1)
    
    while not rospy.is_shutdown():
        key = getKey(settings, KEY_TIMEOUT)
        if key == 'a' or key == 'b' or key == 'c':
            pub_key_action.publish(key)
        elif key == 'q':
            restoreTerminalSettings(settings)
            sys.exit()
        
