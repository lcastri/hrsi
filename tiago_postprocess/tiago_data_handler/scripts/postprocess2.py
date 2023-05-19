from enum import Enum
import pandas as pd
from shapely.geometry import *
import math
import random
import numpy as np


class fNaN_Mode(Enum):
    Interpolation = 0
    Constant = 1
    
    
def check_index(indexes):
    if np.squeeze(indexes).size == 1: 
        return indexes
    else:
        return np.squeeze(indexes)


def fill_missings(df: pd.DataFrame, mode : fNaN_Mode = fNaN_Mode.Constant):
    """
    Fills missing values of the Dataframe

    Args:
        df (pd.DataFrame): complete DataFrame (robot + human)
        mode (fNaN_Mode): Constant | Interpolation
    Returns:
        pd.DataFrame: DataFrame with no NaNs
    """
    if mode == fNaN_Mode.Constant:
        df['h_x'] = df['h_x'].ffill().bfill()
        df['h_y'] = df['h_y'].ffill().bfill()
        df['h_theta'] = df['h_theta'].ffill().bfill()
        df['h_v'] = df['h_v'].apply(lambda l: l if not np.isnan(l) else random.uniform(a=-0.03, b=0.03))
        
    elif mode == fNaN_Mode.Interpolation:

        # h_x
        h_x_notnan = df['h_x'].dropna()
        h_x_bnan = range(0, h_x_notnan.index[0])
        h_x_fnan = range(h_x_notnan.index[-1] + 1, len(df['h_x']))
        h_x_noise = h_x_notnan.diff().std()
        h_x_binter = h_x_notnan[h_x_notnan.index[0]] * np.ones(shape = (len(h_x_bnan),1)) + np.random.normal(0, h_x_noise, size = (len(h_x_bnan),1))
        h_x_finter = h_x_notnan[h_x_notnan.index[-1]] * np.ones(shape = (len(h_x_fnan),1)) + np.random.normal(0, h_x_noise, size = (len(h_x_fnan),1))
        df['h_x'].loc[check_index(h_x_bnan)] = np.squeeze(h_x_binter)
        df['h_x'].loc[check_index(h_x_fnan)] = np.squeeze(h_x_finter)

        # h_y
        h_y_notnan = df['h_y'].dropna()
        h_y_bnan = range(0, h_y_notnan.index[0])
        h_y_fnan = range(h_y_notnan.index[-1] + 1, len(df['h_y']))
        h_y_noise = h_y_notnan.diff().std()
        h_y_average_step = h_y_notnan.diff().mean()
        h_y_binter = [h_y_notnan[h_y_notnan.index[0]] - (t+1)*h_y_average_step + np.random.normal(0, h_y_noise) for t in h_y_bnan]
        h_y_binter.reverse()
        h_y_finter = [h_y_notnan[h_y_notnan.index[-1]] + (t+1)*h_y_average_step + np.random.normal(0, h_y_noise) for t in range(len(h_y_fnan))]
        df['h_y'].loc[check_index(h_y_bnan)] = np.squeeze(h_y_binter)
        df['h_y'].loc[check_index(h_y_fnan)] = np.squeeze(h_y_finter)
        
        # h_theta
        h_theta_notnan = df['h_theta'].dropna()
        h_theta_bnan = range(0, h_theta_notnan.index[0])
        h_theta_fnan = range(h_theta_notnan.index[-1] + 1, len(df['h_theta']))
        h_theta_noise = h_theta_notnan.diff().std()
        h_theta_binter = h_theta_notnan[h_theta_notnan.index[0]] * np.ones(shape = (len(h_theta_bnan),1)) + np.random.normal(0, h_theta_noise, size = (len(h_theta_bnan),1))
        h_theta_finter = h_theta_notnan[h_theta_notnan.index[-1]] * np.ones(shape = (len(h_theta_fnan),1)) + np.random.normal(0, h_theta_noise, size = (len(h_theta_fnan),1))
        df['h_theta'].loc[check_index(h_theta_bnan)] = np.squeeze(h_theta_binter)
        df['h_theta'].loc[check_index(h_theta_fnan)] = np.squeeze(h_theta_finter)
        
        # h_v
        h_v_notnan = df['h_v'].dropna()
        h_v_bnan = range(0, h_v_notnan.index[0])
        h_v_fnan = range(h_v_notnan.index[-1] + 1, len(df['h_v']))
        h_v_noise = h_v_notnan.diff().std()
        h_v_binter = h_v_notnan[h_v_notnan.index[0]] * np.ones(shape = (len(h_v_bnan),1)) + np.random.normal(0, h_v_noise, size = (len(h_v_bnan),1))
        h_v_finter = h_v_notnan[h_v_notnan.index[-1]] * np.ones(shape = (len(h_v_fnan),1)) + np.random.normal(0, h_v_noise, size = (len(h_v_fnan),1))
        df['h_v'].loc[check_index(h_v_bnan)] = np.squeeze(h_v_binter)
        df['h_v'].loc[check_index(h_v_fnan)] = np.squeeze(h_v_finter)
        
    return df

def get_risk(r_old_pos: tuple, h_old_pos: tuple, r_pos: tuple, h_pos: tuple, r_old_vel: float, h_old_vel: float, d_rh_old: float):
    """
    Postprocesses the data and extracts the risk between robot and human

    Args:
        r_old_pos (tuple): t-1 x & y robot position
        h_old_pos (tuple): t-1 x & y human position
        r_pos (tuple): t x & y robot position
        h_pos (tuple): t x & y human position
        r_old_vel (float): t-1 robot velocity
        h_old_vel (float): t-1 human velocity
        d_rh_old (float): t-1 human-robot distance 

    Returns:
        float: risk
    """
    
    # risk = (r_old_vel + h_old_vel) / d_rh_old
    risk = r_old_vel
    
    if d_rh_old < 2.5:
    
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
                      "t_rg": random.uniform(a=-0.05, b=0.05),
                      "d_rh": math.dist([df["r_x"][0], df["r_y"][0]], [df["h_x"][0], df["h_y"][0]]),
                      "risk": random.uniform(a=-0.03, b=0.06),
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
            risk = get_risk(r_old_pos, h_old_pos, r_pos, h_pos, r_old_vel, df["h_v"][t-1], df_new["d_rh"][t-1])
        except:
            risk = 0

        df_new.loc[t] = {"d_rg": math.dist([df["r_x"][t], df["r_y"][t]], [df["g_x"][t], df["g_y"][t]]), 
                         "t_rg": df_new["d_rg"][t-1]/(df["r_v"][t-1] + 0.1) + random.uniform(a=-0.05, b=0.05),
                         "d_rh": math.dist([df["r_x"][t], df["r_y"][t]], [df["h_x"][t], df["h_y"][t]]),
                         "risk": risk + random.uniform(a=-0.03, b=0.06),
                         "theta_rg": math.atan2(df["g_y"][t] - df["r_y"][t], df["g_x"][t] - df["r_x"][t]),
                         "theta_rh": math.atan2(df["h_y"][t] - df["r_y"][t], df["h_x"][t] - df["r_x"][t]), 
                         }
                      
    df_complete = pd.concat([df, df_new], axis = 1)
    return df_complete



if __name__ == '__main__':
    DATA_PATH = r'/home/lcastri/git/tiago_ws/src/hrsi/tiago_postprocess/tiago_postprocess_bringup/data/raw_data'
    NUM_DATASET = 16
    ACTOR = "greta"
    # INTERVENTION = "noaction"
    # INTERVENTION = "decrease"
    INTERVENTION = "increase"
    FILE_EXT = ".csv"
    
    for i in range(NUM_DATASET):
        df_filepath = DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i)
        df_raw = pd.read_csv(df_filepath + "_raw" + FILE_EXT, index_col=0)
        
        # Raw DataFrame filled with constant values
        # df_raw_constant = fill_missings(df_raw, mode = fNaN_Mode.Constant)
        # df_raw_constant.to_csv(df_filepath + "_raw_constant.csv")
        
        # Raw DataFrame filled with interpolated values
        df_raw_inter = fill_missings(df_raw, mode = fNaN_Mode.Interpolation)
        df_raw_inter.to_csv(df_filepath + "_raw_inter.csv")
        
        # Postprocess
        df_final = postprocess(df_raw_inter)
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