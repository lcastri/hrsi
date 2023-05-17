import pandas as pd
from shapely.geometry import *
import math


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
    
    df_new = pd.DataFrame(columns=["d_rg", "t_rg", "d_rh", "risk", "theta_rg", "theta_rh"])
    df_new.loc[0] =  {"d_rg": math.dist([df["r_x"][0], df["r_y"][0]], [df["g_x"][0], df["g_y"][0]]),
                      "t_rg": 0,
                      "d_rh": math.dist([df["r_x"][0], df["r_y"][0]], [df["h_x"][0], df["h_y"][0]]),
                      "risk": 0,
                      "theta_rg": math.atan2(df["g_y"][0] - df["r_y"][0], df["g_x"][0] - df["r_x"][0]),
                      "theta_rh": math.atan2(df["h_y"][0] - df["r_y"][0], df["h_x"][0] - df["r_x"][0]), 
                     }
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

        df_new.loc[t] = {"d_rg": math.dist([df["r_x"][t], df["r_y"][t]], [df["g_x"][t], df["g_y"][t]]), 
                         "t_rg": df_new["d_rg"][t-1]/(df["r_v"][t-1] + 0.1),
                         "d_rh": math.dist([df["r_x"][t], df["r_y"][t]], [df["h_x"][t], df["h_y"][t]]),
                         "risk": risk,
                         "theta_rg": math.atan2(df["g_y"][t] - df["r_y"][t], df["g_x"][t] - df["r_x"][t]),
                         "theta_rh": math.atan2(df["h_y"][t] - df["r_y"][t], df["h_x"][t] - df["r_x"][t]), 
                         }
        #########################################################################################FIXME: new version with only lagged dependency
        # df_new.loc[t] = {"d_rh": math.dist([df["r_x"][t-1], df["r_y"][t-1]], [df["h_x"][t-1], df["h_y"][t-1]]),
        #                  "risk": risk,
        #                  "theta_rg": math.atan2(df["g_y"][t-1] - df["r_y"][t-1], df["g_x"][t-1] - df["r_x"][t-1]),
        #                  "theta_rh": math.atan2(df["h_y"][t-1] - df["r_y"][t-1], df["h_x"][t-1] - df["r_x"][t-1]), 
        #                  }
        #######################################################################################################################################
    df_new['t_rg'].loc[0] = df_new["t_rg"][1]
    df_new['risk'].loc[0] = df_new["risk"][1]
                      
    df_complete = pd.concat([df, df_new], axis = 1)
    return df_complete



if __name__ == '__main__':
    DATA_PATH = r'/home/lucacastri/Git/tiago_ws/src/hrsi/tiago_postprocess/tiago_postprocess_bringup/data'
    NUM_DATASET = 16
    ACTOR = "greta"
    # INTERVENTION = "noaction"
    INTERVENTION = "decrease"
    # INTERVENTION = "increase"
    FILE_EXT = ".csv"
    
    for i in range(NUM_DATASET):
        df_filepath = DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i)
        df_raw = pd.read_csv(df_filepath + "_raw" + FILE_EXT)

        df_final = postprocess(df_raw)
        # df_final.to_csv(df_filepath + "_complete.csv")
        df_final.to_csv(df_filepath + "_causal.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v', 'r_theta',
                                                              'd_rg', 't_rg', 'theta_rg',
                                                              'h_v', 'h_theta', 
                                                              'risk', 'd_rh', 'theta_rh'])
        df_final.to_csv(df_filepath+ "_causal_notheta.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v',
                                                                     'd_rg', 't_rg',
                                                                     'h_v', 
                                                                     'risk', 'd_rh'])
        df_final.to_csv(df_filepath + "_causal_reduced.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v',
                                                                      'd_rg', 'h_v', 'risk', 'd_rh'])