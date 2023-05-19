from enum import Enum
import pandas as pd
from shapely.geometry import *
import math
import numpy as np


D_RH_RISKTHRES = 2.5 # [m]
T_RG_NOISE = 0.03
RISK_NOISE = 0.0175


class fNaN_Mode(Enum):
    Interpolation = 0
    Constant = 1
    
    
def check_index(indexes):
    if np.squeeze(indexes).size == 1: 
        return indexes
    else:
        return np.squeeze(indexes)
    
    
def fmiss(df: pd.DataFrame, column: str):
    notnan = df[column].dropna()
    bnan = range(0, notnan.index[0])
    fnan = range(notnan.index[-1] + 1, len(df[column]))
    std_noise = notnan.diff().std()
    h_x_binter = notnan[notnan.index[0]] * np.ones(shape = (len(bnan),)) + (std_noise * np.random.uniform(-1, 1, size = (len(bnan),)))
    h_x_finter = notnan[notnan.index[-1]] * np.ones(shape = (len(fnan),)) + (std_noise * np.random.uniform(-1, 1, size = (len(fnan),)))
    df.loc[bnan, column] = h_x_binter
    df.loc[fnan, column] = h_x_finter


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
        df['h_v'] = df['h_v'].apply(lambda l: l if not np.isnan(l) else np.random.uniform(-0.07, 0.07))
        
    elif mode == fNaN_Mode.Interpolation:

        # h_x
        fmiss(df, 'h_x')
        
        # h_y
        h_y_notnan = df['h_y'].dropna()
        h_y_bnan = range(0, h_y_notnan.index[0])
        h_y_fnan = range(h_y_notnan.index[-1] + 1, len(df['h_y']))
        h_y_noise = h_y_notnan.diff().std()
        h_y_average_step = h_y_notnan.diff().mean()
        h_y_binter = [h_y_notnan[h_y_notnan.index[0]] - (t+1)*h_y_average_step + (h_y_noise * np.random.uniform(-1, 1)) for t in h_y_bnan]
        h_y_binter.reverse()
        h_y_finter = [h_y_notnan[h_y_notnan.index[-1]] + (t+1)*h_y_average_step + (h_y_noise * np.random.uniform(-1, 1)) for t in range(len(h_y_fnan))]
        df.loc[check_index(h_y_bnan), 'h_y'] = np.squeeze(h_y_binter)
        df.loc[check_index(h_y_fnan), 'h_y'] = np.squeeze(h_y_finter)
        
        # h_theta
        fmiss(df, 'h_theta')
        
        # h_v
        fmiss(df, 'h_v')
        
    return df

# def get_risk(r_old_pos: tuple, h_old_pos: tuple, r_pos: tuple, h_pos: tuple, r_old_vel: float, d_rh_old: float):
#     """
#     Postprocesses the data and extracts the risk between robot and human

#     Args:
#         r_old_pos (tuple): t-1 x & y robot position
#         h_old_pos (tuple): t-1 x & y human position
#         r_pos (tuple): t x & y robot position
#         h_pos (tuple): t x & y human position
#         r_old_vel (float): t-1 robot velocity
#         h_old_vel (float): t-1 human velocity
#         d_rh_old (float): t-1 human-robot distance 

#     Returns:
#         float: risk
#     """
    
#     risk = r_old_vel
#     collision = 0
    
#     # A and B displacements
#     Va = Point(r_pos[0] - r_old_pos[0], r_pos[1] - r_old_pos[1])
#     Vb = Point(h_pos[0] - h_old_pos[0], h_pos[1] - h_old_pos[1])

#     # Relative velocity
#     Vrel = Point(Va.x - Vb.x, Va.y - Vb.y)
    
#     # Cone origin translated by Vb
#     cone_origin = Point(r_old_pos[0] + Vb.x, r_old_pos[1] + Vb.y)
        
#     # A and B position
#     A = Point(r_old_pos[0], r_old_pos[1])
#     B = Point(h_old_pos[0], h_old_pos[1])
                    
#     # Straight line from A to B = r_{a_b}
#     AB = LineString([A, B])
        
#     # PAB _|_ AB passing through B
#     left = AB.parallel_offset(5, 'left')
#     right = AB.parallel_offset(5, 'right')

#     c = left.boundary.geoms[1]
#     d = right.boundary.geoms[0]
#     PAB = LineString([c, d])
                
#     # Straight line perpendicular to r_{a_b} and passing through b
#     B_encumbrance = B.buffer(1.5)
#     if PAB.intersects(B_encumbrance): 
#         inter = PAB.intersection(B_encumbrance).xy
#         inter_l = Point(inter[0][0] + Vb.x, inter[1][0] + Vb.y)
#         inter_r = Point(inter[0][1] + Vb.x, inter[1][1] + Vb.y)
                
#         # Cone
#         cone = Polygon([cone_origin, inter_l, inter_r])
#         P = Point(cone_origin.x + Vrel.x, cone_origin.y + Vrel.y)
#         collision = P.within(cone) and d_rh_old < D_RH_RISKTHRES
#         if collision:
#             time_collision_measure = math.sqrt(Vrel.x**2 + Vrel.y**2)
#             bound1 = LineString([cone_origin, inter_l])
#             bound2 = LineString([cone_origin, inter_r])
#             w_effort_1 = P.distance(bound1)
#             w_effort_2 = P.distance(bound2)
#             w_effort_measure = min(w_effort_1, w_effort_2)           
#             risk = risk + time_collision_measure + w_effort_measure
                
#     return math.exp(risk), collision


# def get_risk2(r_pos: tuple, h_pos: tuple, r_v: float, r_theta: float, h_v: float, h_theta: float, d_rh: float):
    
#     risk = r_v
#     collision = 0
    
#     # Convert absolute velocities to Cartesian velocities
#     Va = Point(r_v * math.cos(r_theta), r_v * math.sin(r_theta))
#     Vb = Point(h_v * math.cos(h_theta), h_v * math.sin(h_theta))
    
#     # Calculate relative velocity vector
#     Vrel = Point(Vb.x - Va.x, Vb.y - Va.y)
    
#     # Cone origin translated by Vb
#     cone_origin = Point(r_pos[0] + Vb.x, r_pos[1] + Vb.y)
        
#     # A and B position
#     A = Point(r_pos[0], r_pos[1])
#     B = Point(h_pos[0], h_pos[1])
                    
#     # Straight line from A to B = r_{a_b}
#     AB = LineString([A, B])
        
#     # PAB _|_ AB passing through B
#     left = AB.parallel_offset(5, 'left')
#     right = AB.parallel_offset(5, 'right')

#     c = left.boundary.geoms[1]
#     d = right.boundary.geoms[0]
#     PAB = LineString([c, d])
                
#     # Straight line perpendicular to r_{a_b} and passing through b
#     B_encumbrance = B.buffer(1.5)
#     if PAB.intersects(B_encumbrance): 
#         inter = PAB.intersection(B_encumbrance).xy
#         inter_l = Point(inter[0][0] + Vb.x, inter[1][0] + Vb.y)
#         inter_r = Point(inter[0][1] + Vb.x, inter[1][1] + Vb.y)
                
#         # Cone
#         cone = Polygon([cone_origin, inter_l, inter_r])
#         P = Point(cone_origin.x + Vrel.x, cone_origin.y + Vrel.y)
#         collision = P.within(cone) and d_rh < D_RH_RISKTHRES
#         if collision:
#             time_collision_measure = math.sqrt(Vrel.x**2 + Vrel.y**2)
#             bound1 = LineString([cone_origin, inter_l])
#             bound2 = LineString([cone_origin, inter_r])
#             w_effort_1 = P.distance(bound1)
#             w_effort_2 = P.distance(bound2)
#             w_effort_measure = min(w_effort_1, w_effort_2)           
#             risk = risk + time_collision_measure + w_effort_measure
                
#     return math.exp(risk), collision


def get_risk(r_vel, r_theta, h_vel, h_theta, d_rh):
    
    # Convert absolute velocities to Cartesian velocities
    r_dvel = [r_vel * math.cos(r_theta), r_vel * math.sin(r_theta)]
    h_dvel = [h_vel * math.cos(h_theta), h_vel * math.sin(h_theta)]
    
    # Calculate relative velocity vector
    relative_vel = [h_dvel[0] - r_dvel[0], h_dvel[1] - r_dvel[1]]
    
    # Calculate time to collision (TTC)
    ttc = d_rh / math.sqrt(relative_vel[0]**2 + relative_vel[1]**2)

    return 1 / ttc


def postprocess(df: pd.DataFrame):
    """
    Adds distance human-robot, risk, angle robot-goal, angle robot-human to the dataframe

    Args:
        df (pd.DataFrame): dataframe to complete

    Returns:
        DataFrame: completed dataframe
    """
    
    df_new = pd.DataFrame(columns=["d_rg", "t_rg", "d_rh", "collision", "risk", "theta_rg", "theta_rh"])
    df_new.loc[0] =  {"d_rg": math.dist([df["r_x"][0], df["r_y"][0]], [df["g_x"][0], df["g_y"][0]]),
                      "t_rg": np.random.uniform(-T_RG_NOISE, T_RG_NOISE),
                      "d_rh": math.dist([df["r_x"][0], df["r_y"][0]], [df["h_x"][0], df["h_y"][0]]),
                      "risk": np.random.uniform(-RISK_NOISE, RISK_NOISE),
                      "theta_rg": math.atan2(df["g_y"][0] - df["r_y"][0], df["g_x"][0] - df["r_x"][0]),
                      "theta_rh": math.atan2(df["h_y"][0] - df["r_y"][0], df["h_x"][0] - df["r_x"][0]), 
                     }
    for t in range(1, len(df)):
        # r_old_pos = (df["r_x"][t-1], df["r_y"][t-1])
        # h_old_pos = (df["h_x"][t-1], df["h_y"][t-1])
        # r_old_vel = df["r_v"][t-1]
        # r_pos = (df["r_x"][t], df["r_y"][t])
        # h_pos = (df["h_x"][t], df["h_y"][t])
        # risk, _ = get_risk(r_old_pos, h_old_pos, r_pos, h_pos, r_old_vel, df_new["d_rh"][t-1])
        risk = get_risk(df["r_v"][t-1], df["r_theta"][t-1], df["h_v"][t-1], df["h_theta"][t-1], df_new["d_rh"][t-1])
        # risk, _ = get_risk2(r_old_pos, h_old_pos, df["r_v"][t-1], df["r_theta"][t-1], df["h_v"][t-1], df["h_theta"][t-1], df_new["d_rh"][t-1])

        df_new.loc[t] = {"d_rg": math.dist([df["r_x"][t], df["r_y"][t]], [df["g_x"][t], df["g_y"][t]]), 
                         "t_rg": df_new["d_rg"][t-1]/(df["r_v"][t-1]) + np.random.uniform(-T_RG_NOISE, T_RG_NOISE),
                         "d_rh": math.dist([df["r_x"][t], df["r_y"][t]], [df["h_x"][t], df["h_y"][t]]),
                         "risk": risk +  np.random.uniform(-RISK_NOISE, RISK_NOISE),
                         "theta_rg": math.atan2(df["g_y"][t] - df["r_y"][t], df["g_x"][t] - df["r_x"][t]),
                         "theta_rh": math.atan2(df["h_y"][t] - df["r_y"][t], df["h_x"][t] - df["r_x"][t]), 
                         }
                      
    df_complete = pd.concat([df, df_new], axis = 1)
    return df_complete



if __name__ == '__main__':
    RAWDATA_PATH = r'/home/lucacastri/git/tiago_ws/src/hrsi/tiago_postprocess/tiago_postprocess_bringup/raw_data'
    DATA_PATH = r'/home/lucacastri/git/tiago_ws/src/hrsi/tiago_postprocess/tiago_postprocess_bringup/data'
    NUM_DATASET = 16
    ACTOR = "greta"
    # INTERVENTION = "noaction"
    INTERVENTION = "decrease"
    # INTERVENTION = "increase"
    # INTERVENTION = "hlr"
    # INTERVENTION = "hud"
    FILE_EXT = ".csv"
    
    for i in range(NUM_DATASET):
        df_filepath = RAWDATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i)
        df_savepath = DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i)
        df_raw = pd.read_csv(df_filepath + "_raw" + FILE_EXT, index_col=0)
        
        # Raw DataFrame filled with constant values
        # df_raw_constant = fill_missings(df_raw, mode = fNaN_Mode.Constant)
        # df_raw_constant.to_csv(df_savepath + "_raw_constant.csv")
        
        # Raw DataFrame filled with interpolated values
        df_raw_inter = fill_missings(df_raw, mode = fNaN_Mode.Interpolation)
        # df_raw_inter.to_csv(df_savepath + "_raw_inter.csv")
        
        # Postprocess
        df_final = postprocess(df_raw_inter)
        df_final.to_csv(df_savepath + "_causal.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v', 'r_theta',
                                                              'd_rg', 't_rg', 'theta_rg',
                                                              'h_v', 'h_theta', 
                                                              'risk', 'd_rh', 'theta_rh'])
        df_final.to_csv(df_savepath+ "_causal_notheta.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v',
                                                                     'd_rg', 't_rg',
                                                                     'h_v', 
                                                                     'risk', 'd_rh'])
        df_final.to_csv(df_savepath + "_causal_reduced.csv", columns=['r_v_h1', 'r_v_h2', 'r_v_t', 'r_v',
                                                                      'd_rg', 'h_v', 'risk', 'd_rh'])