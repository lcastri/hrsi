#!/usr/bin/env python

from geometry_msgs.msg import Twist
import rospy

NODE_NAME = "bridge_vel"
NODE_RATE = 10
SCALAR_FACTOR = 0.2

class BridgeVel():
    """
    Class handling different pre-defined movements of the TIAGo robot
    """
    
    def __init__(self):
        """
        Class constructor. Init publisher and subscriber
        """
        # Init subscriber & publisher
        self.pub_cmd_vel = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 10)
        self.sub_tmp_cmd_vel = rospy.Subscriber('/mobile_base_controller/tmp_cmd_vel', Twist, self.cb_vel)
               
        
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
    
    action_controller = BridgeVel()
        
    rospy.spin()