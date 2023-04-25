#!/usr/bin/env python

from enum import Enum
import math
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import message_filters
import rospy
import pandas as pd
from shapely.geometry import *
from people_msgs.msg import People
import tf


FILENAME = str(rospy.get_param("/tiago_data_handler/bagname"))
DATAPATH = str(rospy.get_param("/tiago_data_handler/datapath"))
NODE_NAME = 'tiago_data_handler'
NODE_RATE = 100 #Hz


class Goal(Enum):
    X = (-0.945, 4.376)
    Y = (5.531, 6.961)
    
GOAL = None


def get_risk(r_old_pos: tuple, h_old_pos: tuple, r_pos: tuple, h_pos: tuple, r_old_vel: float):
    """
    Postprocesses the data and extracts the risk between robot and human

    Args:
        r_old_pos (tuple): t-1 x & y robot position
        h_old_pos (tuple): t-1 x & y human position
        r_pos (tuple): t x & y robot position
        h_pos (tuple): t x & y human position
        r_old_vel (float): t-1 robot velocity

    Returns:
        float: risk
    """
    
    risk = r_old_vel
    
    # A and B displacements
    Va = Point(r_pos[0] - r_old_pos[0], r_pos[1] - r_old_pos[1])
    Vb = Point(h_pos[0] - h_old_pos[0], h_pos[1] - h_old_pos[1])

    # Relative velocity
    Vrel = Point(Va.x - Vb.x, Va.y - Vb.y)

    # Cone origin translated by Vb
    cone_origin = Point(r_old_pos[0] + Vb.x, r_old_pos[1] + Vb.y)
    
    # A and B position
    A = Point(r_old_pos[0], r_old_pos[1])
    B = Point(h_old_pos[0], h_old_pos[1])
                
    # Straight line from A to B = r_{a_b}
    AB = LineString([A, B])
       
    # PAB _|_ AB passing through B
    left = AB.parallel_offset(5, 'left')
    right = AB.parallel_offset(5, 'right')

    c = left.boundary.geoms[1]
    d = right.boundary.geoms[0]
    PAB = LineString([c, d])
            
    # Straight line perpendicular to r_{a_b} and passing through b
    B_encumbrance = B.buffer(1.5)
    if PAB.intersects(B_encumbrance): 
        inter = PAB.intersection(B_encumbrance).xy
        inter_l = Point(inter[0][0] + Vb.x, inter[1][0] + Vb.y)
        inter_r = Point(inter[0][1] + Vb.x, inter[1][1] + Vb.y)
            
        # Cone
        cone = Polygon([cone_origin, inter_l, inter_r])
        P = Point(cone_origin.x + Vrel.x, cone_origin.y + Vrel.y)
        collision = P.within(cone)
        if collision:
            time_collision_measure = math.sqrt(Vrel.x**2 + Vrel.y**2)
            bound1 = LineString([cone_origin, inter_l])
            bound2 = LineString([cone_origin, inter_r])
            w_effort_1 = P.distance(bound1)
            w_effort_2 = P.distance(bound2)
            w_effort_measure = min(w_effort_1, w_effort_2)           
            risk = risk + time_collision_measure + w_effort_measure
                
    return math.exp(risk)
    
    
def postprocess(df: pd.DataFrame):
    """
    Adds distance human-robot, risk, angle robot-goal, angle robot-human to the dataframe

    Args:
        df (pd.DataFrame): dataframe to complete

    Returns:
        DataFrame: completed dataframe
    """
    
    df_new = pd.DataFrame(columns=["d_rh", "risk", "theta_rg", "theta_rh"])
    for t in range(1, len(df)):
        r_old_pos = (df["r_x"][t-1], df["r_y"][t-1])
        h_old_pos = (df["h_x"][t-1], df["h_y"][t-1])
        r_old_vel = df["r_v"][t-1]
        r_pos = (df["r_x"][t], df["r_y"][t])
        h_pos = (df["h_x"][t], df["h_y"][t])
        try:
            risk = get_risk(r_old_pos, h_old_pos, r_pos, h_pos, r_old_vel)
        except:
            risk = 0
        df_new.loc[t] = {"d_rh": math.dist([df["r_x"][t], df["r_y"][t]], [df["h_x"][t], df["h_y"][t]]),
                         "risk": risk,
                         "theta_rg": math.atan2(GOAL[1] - df["r_y"][t], GOAL[0] - df["r_x"][t]),
                         "theta_rh": math.atan2(df["h_y"][t] - df["r_y"][t], df["h_x"][t] - df["r_x"][t]), 
                         }
    
    df_new.loc[0] =  {"d_rh": math.dist([df["r_x"][0], df["r_y"][0]], [df["h_x"][0], df["h_y"][0]]),
                      "risk": df_new["risk"][1],}
                      
    df_complete = pd.concat([df, df_new], axis = 1)
    return df_complete


def get_2DPose(p: PoseWithCovarianceStamped):
    """
    Extracts x, y and theta from pose

    Args:
        p (PoseWithCovarianceStamped): pose

    Returns:
        tuple: x, y, theta
    """
    x = p.pose.pose.position.x
    y = p.pose.pose.position.y
    
    q = (
        p.pose.pose.orientation.x,
        p.pose.pose.orientation.y,
        p.pose.pose.orientation.z,
        p.pose.pose.orientation.w
    )
    
    m = tf.transformations.quaternion_matrix(q)
    _, _, yaw = tf.transformations.euler_from_matrix(m)
    return x, y, yaw
            

class DataHandler():
    """
    Class handling data
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.df_robot = pd.DataFrame(columns=['g_x', 'g_y', 'r_x', 'r_y', 'r_theta', 'r_v_h1', 'r_v_h2', 'r_v_t', 'r_v', 'r_omega', 'd_rg', 't_rg'])
        self.people_dict = dict()
        # self.df = pd.DataFrame(columns = ['vel_h1', 'vel_h2', 'vel_t', 'vel_r', 'omega_r', 'd_hr', 'risk', 'v_h', 'theta_hr'])
        
        # Head subscriber
        self.sub_head_state = message_filters.Subscriber("/head_controller/state", JointTrajectoryControllerState)
        
        # Torso subscriber
        self.sub_torso_state = message_filters.Subscriber('/torso_controller/state', JointTrajectoryControllerState)
        
        # Base subscriber
        self.sub_cmd_vel = message_filters.Subscriber('/mobile_base_controller/cmd_vel', Twist)
                
        # Robot pose subscriber
        self.sub_robot_pos = message_filters.Subscriber('/robot_pose', PoseWithCovarianceStamped)
                
        # People subscriber
        self.sub_person_pos = message_filters.Subscriber('/people_tracker/people', People)
        
        # Init synchronizer and assigning a callback 
        self.ats = message_filters.ApproximateTimeSynchronizer([self.sub_head_state, 
                                                                self.sub_torso_state, 
                                                                self.sub_cmd_vel, 
                                                                self.sub_robot_pos, 
                                                                self.sub_person_pos], 
                                                                queue_size = 100, slop = 1,
                                                                allow_headerless = True)
        self.ats.registerCallback(self.cb_handle_data)
        
        
    def people_handler(self, people: list, index: int):
        """
        Extracts people state (x, y, vel)

        Args:
            people (list): list of people
            index (int): current time step
        """
        for p in people:
            # Velocity
            vel = math.sqrt(p.velocity.x**2 + p.velocity.y**2)
            
            # Orientation
            theta = math.atan2(p.velocity.y, p.velocity.x)
            theta = math.fmod(theta, 2*math.pi)
            if theta < 0:
                theta += 2*math.pi
            
            # Position
            if vel >= 0.75:
                if p.name not in self.people_dict:
                    self.people_dict[p.name] = pd.DataFrame(columns=["h_x", "h_y", "h_v", "h_theta"])
                self.people_dict[p.name].loc[index] = {"h_x": p.position.x,
                                                       "h_y": p.position.y,
                                                       "h_v": vel,
                                                       "h_theta": theta}
                
                
    def cb_handle_data(self, head_state: JointTrajectoryControllerState, 
                             torso_state: JointTrajectoryControllerState, 
                             robot_vel: Twist, 
                             robot_pose: PoseWithCovarianceStamped,
                             people: People):
        """
        Synchronized callback

        Args:
            head_state (JointTrajectoryControllerState): robot head state
            torso_state (JointTrajectoryControllerState): robot torso state
            robot_vel (Twist): robot base velocity
            robot_pose (PoseWithCovarianceStamped): robot pose
            people (People): tracked people
        """
        
        # Robot 2D pose (x, y, theta)
        r_x, r_y, r_theta = get_2DPose(robot_pose)
        
        # Head motors 1 & 2 velocities
        head1_vel = head_state.actual.velocities[0]
        head2_vel = head_state.actual.velocities[1]
        
        # Torso velocity
        torso_vel = torso_state.actual.velocities[0]
        
        # Base linear & angular velocity
        base_vel = robot_vel.linear.x
        base_ang_vel = robot_vel.angular.z
        
        # Distance Robot-current_goal & time to reach current_goal
        d_rg = math.dist([r_x, r_y], [GOAL[0], GOAL[1]])
        t_rg = d_rg/(base_vel + 0.001)
        
        # appending new data row in robot Dataframe
        self.df_robot.loc[len(self.df_robot)] = {'g_x': GOAL[0], 'g_y': GOAL[1],
                                                 'r_x': r_x, 'r_y': r_y, 'r_theta': r_theta,
                                                 'r_v_h1': head1_vel, 'r_v_h2': head2_vel, 
                                                 'r_v_t': torso_vel, 
                                                 'r_v': base_vel, 'r_omega': base_ang_vel,
                                                 'd_rg': d_rg, 't_rg': t_rg,
                                                 }
        
        self.people_handler(people.people, len(self.df_robot))      
        
            

if __name__ == '__main__':
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)

    # Goal definition
    GOAL = Goal.Y.value if int(FILENAME[-1]) % 2 == 0 else Goal.X.value
    
    data_handler = DataHandler()
        
    rospy.spin()
            
    if data_handler.people_dict:
        # FIXME: this take the tracked person with longest time-series. 
        # It should be changed to take the correct person. 
        len_tmp = {}
        for p in data_handler.people_dict:
            len_tmp[p] = len(data_handler.people_dict[p])
        p = max(len_tmp, key=len_tmp.get)
        ###############################################################
            
        df_complete = pd.concat([data_handler.df_robot, data_handler.people_dict[p]], axis = 1)
        df_complete = df_complete.ffill().bfill()
        
        df_final = postprocess(df_complete)
        df_final.to_csv(DATAPATH + "/" + FILENAME + "_complete.csv")
        df_final.to_csv(DATAPATH + "/" + FILENAME + "_causal.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v', 'r_theta',
                                                                            'd_rg', 't_rg', 'theta_rg',
                                                                            'h_v', 'h_theta', 
                                                                            'risk', 'd_rh', 'theta_rh'])