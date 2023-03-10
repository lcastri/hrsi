from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from constants import *
import rospy


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