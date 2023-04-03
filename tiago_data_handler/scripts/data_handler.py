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


FILENAME = str(rospy.get_param("/tiago_data_handler/bagname"))
DATAPATH = str(rospy.get_param("/tiago_data_handler/datapath"))
NODE_NAME = 'tiago_data_handler'
NODE_RATE = 100 #Hz


class Goal(Enum):
    X = (-0.945, 4.376)
    Y = (5.531, 6.961)
    
GOAL = None
            

class DataHandler():
    """
    Class handling data
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.df_robot = pd.DataFrame(columns=['x_g', 'y_g', 'x_r', 'y_r', 'vel_h1', 'vel_h2', 'vel_t', 'vel_r', 'omega_r', 'd_rg', 't_rg'])
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
        for p in people:
            vel = math.sqrt(p.velocity.x**2 + p.velocity.y**2)
            if vel >= 0.75:
                if p.name not in self.people_dict:
                    self.people_dict[p.name] = pd.DataFrame(columns=["x_h", "y_h", "vel_h"])
                self.people_dict[p.name].loc[index] = {"x_h":p.position.x,
                                                       "y_h":p.position.y,
                                                       "vel_h":vel}
                
                
    def cb_handle_data(self, head_state: JointTrajectoryControllerState, 
                             torso_state: JointTrajectoryControllerState, 
                             robot_vel: Twist, 
                             robot_pose: PoseWithCovarianceStamped,
                             people: People):
        
        # robot_pos_x = robot_pose.pose.pose.position.x
        # robot_pos_y = robot_pose.pose.pose.position.y
        # person_pos_x = people.people
        # person_pos_y = people.pose.position.y
        
        # # Update robot state
        # self.robot.update_pos(robot_pos_x, robot_pos_y)
        # self.robot.update_vel(robot_vel)
        
        # # Update human state
        # self.human.update_pos(person_pos_x, person_pos_y)
        # self.human.update_vel(human_vel) #FIXME: find the human velocity coming from tracker
        
        # self.df_pos.loc[len(self.df_pos)] = {'x_g':GOAL[0],
        #                                      'y_g':GOAL[1],
        #                                      'x_r':robot_pos_x, 
        #                                      'y_r':robot_pos_y, 
        #                                      'x_h':person_pos_x, 
        #                                      'y_h':person_pos_y}
        
        # head1_vel = head_state.actual.velocities[0]
        # head2_vel = head_state.actual.velocities[1]
        # torso_vel = torso_state.actual.velocities[0]
        # base_vel = robot_vel.linear.x
        # base_ang_vel = robot_vel.angular.z
        # d_hr = math.dist([robot_pos_x, robot_pos_y], [person_pos_x, person_pos_y])
        # risk = self.get_risk()
        # # #TODO: compute v_h
        # # #TODO: compute risk
        # # #TODO: compute angle 
        # self.df.loc[len(self.df)] = {'vel_h1': head1_vel, 
        #                              'vel_h2': head2_vel, 
        #                              'vel_t': torso_vel, 
        #                              'vel_r': base_vel, 
        #                              'omega_r': base_ang_vel,
        #                              'd_hr': d_hr,
        #                              'risk': risk,
        #                              } #TODO: add the other vars
        
        robot_pos_x = robot_pose.pose.pose.position.x
        robot_pos_y = robot_pose.pose.pose.position.y
        
        head1_vel = head_state.actual.velocities[0]
        head2_vel = head_state.actual.velocities[1]
        torso_vel = torso_state.actual.velocities[0]
        base_vel = robot_vel.linear.x
        base_ang_vel = robot_vel.angular.z
        d_rg = math.dist([robot_pos_x, robot_pos_y], [GOAL[0], GOAL[1]])
        t_rg = d_rg/(base_vel + 0.001)
        self.df_robot.loc[len(self.df_robot)] = {'x_g':GOAL[0],
                                                 'y_g':GOAL[1],
                                                 'x_r':robot_pos_x, 
                                                 'y_r':robot_pos_y,
                                                 'vel_h1': head1_vel, 
                                                 'vel_h2': head2_vel, 
                                                 'vel_t': torso_vel, 
                                                 'vel_r': base_vel, 
                                                 'omega_r': base_ang_vel,
                                                 'd_rg': d_rg,
                                                 't_rg': t_rg,
                                                 }
        self.people_handler(people.people, len(self.df_robot))      
        
        
def get_risk(r_old_pos, h_old_pos, r_pos, h_pos, r_old_vel):     
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
    df_dhr = pd.DataFrame(columns=["d_rh"])
    df_risk = pd.DataFrame(columns=["risk"])
    for t in range(1, len(df)):
        r_old_pos = (df["x_r"][t-1], df["y_r"][t-1])
        h_old_pos = (df["x_h"][t-1], df["y_h"][t-1])
        r_old_vel = df["vel_r"][t-1]
        r_pos = (df["x_r"][t], df["y_r"][t])
        h_pos = (df["x_h"][t], df["y_h"][t])
        try:
            risk = get_risk(r_old_pos, h_old_pos, r_pos, h_pos, r_old_vel)
        except:
            risk = 0
        df_risk.loc[t] = {"risk": risk}
        df_dhr.loc[t] = {"d_rh": math.dist([df["x_r"][t], df["y_r"][t]], [df["x_h"][t], df["y_h"][t]])}
    
    df_dhr.loc[0] =  math.dist([df["x_r"][0], df["y_r"][0]], [df["x_h"][0], df["y_h"][0]])
    df_risk.loc[0] = df_risk.loc[1]
    df_complete = pd.concat([df, df_dhr, df_risk], axis = 1)
    return df_complete
            

if __name__ == '__main__':
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)

    # Goal definition
    GOAL = Goal.Y.value if int(FILENAME[-1]) % 2 == 0 else Goal.X.value
    
    data_handler = DataHandler()
        
    rospy.spin()
            
    len_tmp = {}
    for p in data_handler.people_dict:
        len_tmp[p] = len(data_handler.people_dict[p])

    p = max(len_tmp, key=len_tmp.get)
        
    df_complete = pd.concat([data_handler.df_robot, data_handler.people_dict[p]], axis = 1)
    df_complete = df_complete.ffill().bfill()
    
    df_final = postprocess(df_complete)
    df_final.to_csv(DATAPATH + "/" + FILENAME + "_complete.csv")
    df_final.to_csv(DATAPATH + "/" + FILENAME + "_causal.csv", columns=['vel_h1', 'vel_h2', 'vel_t', 'vel_r', 'omega_r', 'd_rg', 't_rg', 'vel_h', 'risk', 'd_rh'])