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
INTERVENTION = ["decrease"]
FILE_EXT = ".csv"


if __name__ == '__main__':
    
    li = []
    # FIXME: to use with the complete csv
    # all_files = [DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i) + "_causal" + FILE_EXT for i in range(NUM_DATASET)]
    # colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"r_{\theta}", r"d_{rg}", r"t_{rg}", r"\theta_{rg}", r"h_v", r"h_{theta}", r"risk", r"d_{rh}", r"\theta_{rh}"] 
    
    # FIXME: to use with the onlyorient csv
    # all_files = [DATA_PATH + "/" + ACTOR + "_" + i + "_" + str(d) + "_causal_onlyorient" + FILE_EXT for i in INTERVENTION for d in range(NUM_DATASET)]
    # colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"r_{\theta}", r"d_{rg}", r"t_{rg}", r"h_v", r"h_{\theta}", r"risk", r"d_{rh}"]
    
    # FIXME: to use with the notheta csv
    all_files = [DATA_PATH + "/" + ACTOR + "_" + i + "_" + str(d) + "_causal_notheta" + FILE_EXT for i in INTERVENTION for d in range(NUM_DATASET)]
    colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"d_{rg}", r"t_{rg}", r"h_v", r"risk", r"d_{rh}"]
    
    # FIXME: to use with the reduced csv
    # all_files = [DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i) + "_causal_reduced" + FILE_EXT for i in range(NUM_DATASET)]
    # colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"d_{rg}", r"h_v", r"risk", r"d_{rh}"]
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename, index_col = [0], names=colnames, header = 0)
            li.append(df)
        except:
            continue
    frame = pd.concat(li, axis = 0, ignore_index = True)
    
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = Data(frame)
    df.plot_timeseries()
    start = time()
    fpcmci = FPCMCI(df, 
                    f_alpha = f_alpha,
                    pcmci_alpha = pcmci_alpha,
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True,
                    resfolder = ACTOR + "_" + "_".join(INTERVENTION) + "_ttc_uninoise_03trg_0175risk_05alpha_NEW")
    
    fpcmci_res, causal_model = fpcmci.run()
    elapsed_FPCMCI = time() - start
    print(str(timedelta(seconds = elapsed_FPCMCI)))
    
    
    colors = {r"r_{vh1}" : "orangered", 
              r"r_{vh2}" : "orangered", 
              r"r_{vt}" : "orangered", 
              r"r_v" : "orangered", 
              r"d_{rg}": "dodgerblue", 
              r"t_{rg}": "dodgerblue", 
              r"h_v": "dodgerblue", 
              r"risk": "dodgerblue", 
              r"d_{rh}" : "dodgerblue"}
    nodes_color = {'$' + key + '$': colors[key] for key in fpcmci_res}
    fpcmci.dag(label_type = LabelType.NoLabels, node_layout = 'circular', node_color = nodes_color)