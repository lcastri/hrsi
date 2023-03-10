from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from constants import *
import rospy


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