from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from constants import *
import rospy


def create_torso_msg(where: torso_movements):
    torso_cmd = JointTrajectory()
       
    # Joint name to move
    torso_cmd.joint_names = [J_TORSO]
    
    # Joint trajectory point to reach
    point = JointTrajectoryPoint
    point.positions = [J_TORSO_UP if where == torso_movements.UP else J_TORSO_DOWN]
    point.velocities = []
    point.accelerations = []
    point.effort = []
    point.time_from_start = rospy.Duration(J_TORSO_TIME) #secs
    torso_cmd.points = [point]
    
    return torso_cmd