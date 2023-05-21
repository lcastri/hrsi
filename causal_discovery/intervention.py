# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# matplotlib inline     
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
from scipy.stats import gaussian_kde

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb

from tigramite.models import Models
from tigramite.causal_effects import CausalEffects

import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


# graph =  np.array([[['', '-->', ''],
#                     ['', '', ''],
#                     ['', '', '']],
#                    [['', '-->', ''],
#                     ['', '-->', ''],
#                     ['-->', '', '-->']],
#                    [['', '', ''],
#                     ['<--', '', ''],
#                     ['', '-->', '']]], dtype='<U3')

# X = [(1,-2)]
# Y = [(2,0)]
# causal_effects = CausalEffects(graph, graph_type='stationary_dag', X=X, Y=Y, S=None, 
#                                hidden_variables=None, 
#                             verbosity=1)
# var_names = ['$X^0$', '$X^1$', '$X^2$']

# opt = causal_effects.get_optimal_set()
# print("Oset = ", [(var_names[v[0]], v[1]) for v in opt])
# special_nodes = {}
# for node in causal_effects.X:
#     special_nodes[node] = 'red'
# for node in causal_effects.Y:
#     special_nodes[node] = 'blue'
# for node in opt:
#     special_nodes[node] = 'orange'
# for node in causal_effects.M:
#     special_nodes[node] = 'lightblue'

    
# tp.plot_time_series_graph(graph = causal_effects.graph,
#         var_names=var_names, 
# #         save_name='Example.pdf',
#         figsize = (8, 4),
#         special_nodes=special_nodes
#         ); plt.show()


# coeff = .5
# def nonlin_f(x): return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))
# links_coeffs = {
#                 0: [((0, -1), coeff, nonlin_f), ((1, -1), coeff, nonlin_f)], 
#                 1: [((1, -1), coeff, nonlin_f),], 
#                 2: [((2, -1), coeff, nonlin_f), ((1, 0), coeff, nonlin_f), ((1,-2), coeff, nonlin_f)],
#                 }
# # Observational data
# T = 10000
# data, nonstat = toys.structural_causal_process(links_coeffs, T=T, noises=None, seed=7)
# dataframe = pp.DataFrame(data)

# # Fit causal effect model from observational data
# causal_effects.fit_total_effect(
#         dataframe=dataframe, 
#         estimator=KNeighborsRegressor(),
#         adjustment_set='optimal',
#         )

# intervention_data = 1.*np.ones((1, 1))
# y1 = causal_effects.predict_total_effect( 
#         intervention_data=intervention_data,
#         )

# intervention_data = 0.*np.ones((1, 1))
# y2 = causal_effects.predict_total_effect( 
#         intervention_data=intervention_data,
#         )

# beta = (y1 - y2)
# print("Causal effect is %.2f" %(beta))





from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
from time import time
from datetime import timedelta
import pandas as pd


DATA_PATH = r'/home/lucacastri/git/tiago_ws/src/hrsi/tiago_postprocess/tiago_postprocess_bringup/data'
NUM_DATASET = 16
ACTOR = "greta"
OBSERVATION_DATA = "noaction"
INTERVETION1_DATA = "decrease"
INTERVETION2_DATA = "increase"
FILE_EXT = ".csv"

def get_data(data):
    li = []
    colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"d_{rg}", r"t_{rg}", r"h_v", r"risk", r"d_{rh}"]

    files = [DATA_PATH + "/" + ACTOR + "_" + data + "_" + str(d) + "_causal_notheta" + FILE_EXT for d in range(NUM_DATASET)]
    for filename in files:
        try:
            df = pd.read_csv(filename, index_col = [0], names=colnames, header = 0)
            li.append(df)
        except:
            continue
    return pd.concat(li, axis = 0, ignore_index = True)


if __name__ == '__main__':
    
    df_obs = get_data(OBSERVATION_DATA)
    df_int1 = get_data(INTERVETION1_DATA)
    df_int2 = get_data(INTERVETION2_DATA)   
    
    true_effect = {r"d_{rg}" : 0,
                   r"t_{rg}" : 0,
                   r"risk" : 0}
    
    true_effect[r"d_{rg}"] = (df_int1[r"d_{rg}"] - df_int2[r"d_{rg}"]).mean(skipna=True)
    true_effect[r"t_{rg}"] = (df_int1[r"t_{rg}"] - df_int2[r"t_{rg}"]).mean(skipna=True)
    true_effect[r"risk"] = (df_int1[r"risk"] - df_int2[r"risk"]).mean(skipna=True)
    
    for k, v in true_effect.items(): print(str(k) + " : " + str(v))
    
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = Data(df_obs)
    # df.plot_timeseries()
    start = time()
    fpcmci = FPCMCI(df, 
                    f_alpha = f_alpha,
                    pcmci_alpha = pcmci_alpha,
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True)
    
    fpcmci_res, causal_model = fpcmci.run()
    # elapsed_FPCMCI = time() - start
    # print(str(timedelta(seconds = elapsed_FPCMCI)))

    
    # colors = {r"r_{vh1}" : "orangered", 
    #           r"r_{vh2}" : "orangered", 
    #           r"r_{vt}" : "orangered", 
    #           r"r_v" : "orangered", 
    #           r"d_{rg}": "dodgerblue", 
    #           r"t_{rg}": "dodgerblue", 
    #           r"h_v": "dodgerblue", 
    #           r"risk": "dodgerblue", 
    #           r"d_{rh}" : "dodgerblue"}
    # nodes_color = {'$' + key + '$': colors[key] for key in fpcmci_res}
    # fpcmci.dag(label_type = LabelType.NoLabels, node_layout = 'circular', node_color = nodes_color)
    
    graph = fpcmci.validator.result['graph']
    X = [(3, -1)]
    # Y = [(4, 0), (5, 0), (7, 0)]
    Y = [(7, 0)]
    causal_effects = CausalEffects(graph, graph_type = 'stationary_dag', X = X, Y = Y, S = None, 
                                   hidden_variables = None, 
                                   verbosity = 1)
    
    causal_effects.fit_total_effect(dataframe = OBSERVATION_DATA, 
                                    estimator = GaussianProcessRegressor(),
                                    adjustment_set = 'optimal',
                                    conditional_estimator = None,  
                                    data_transform = None,
                                    mask_type = None)
    
    y1 = causal_effects.predict_total_effect(intervention_data = df_int1[r"r_v"])
    print("y1 = ", y1)

    y2 = causal_effects.predict_total_effect(intervention_data = df_int2[r"r_v"])
    print("y2 = ", y2)