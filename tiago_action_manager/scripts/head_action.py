from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from constants import *
import rospy


def create_head_msg(where: head_movements):
    head_cmd = JointTrajectory()
    # REMINDER: 
    # J_HEAD_1 left & right
    # J_HEAD_2 up & down
    
    # Joint name to move
    head_cmd.joint_names = [J_HEAD_2, J_HEAD_1]
    
    if where == head_movements.LEFT or where == head_movements.RIGHT:
        # Joint trajectory point to reach
        point = JointTrajectoryPoint
        point.positions = [0.0, J_HEAD_1_LEFT if where == head_movements.LEFT else J_HEAD_1_RIGHT]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_1_TIME) #secs
        head_cmd.points = [point]
        
    elif where == head_movements.UP or where == head_movements.DOWN:  
        # Joint trajectory point to reach
        point = JointTrajectoryPoint
        point.positions = [J_HEAD_2_UP if where == head_movements.UP else J_HEAD_2_DOWN, 0.0]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = [0.0, 0.0]
        point.time_from_start = rospy.Duration(J_HEAD_2_TIME) #secs
        head_cmd.points = [point]
    
    return head_cmd