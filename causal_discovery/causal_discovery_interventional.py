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
# INTERVENTION = ["increase"]
INTERVENTION = ["decrease", "increase"]
# INTERVENTION = ["noaction", "decrease", "increase"]
FILE_EXT = ".csv"


if __name__ == '__main__':
    
    li = []
    
    all_files = [DATA_PATH + "/" + ACTOR + "_" + i + "_" + str(d) + "_causal_notheta" + FILE_EXT for i in INTERVENTION for d in range(NUM_DATASET)]
    colnames = [r"r_{vh1}", r"r_{vh2}", r"r_{vt}", r"r_v", r"d_{rg}", r"t_{rg}", r"h_v", r"risk", r"d_{rh}"]
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename, index_col = [0], names=colnames, header = 0)
            li.append(df)
        except:
            continue
    frame = pd.concat(li, axis = 0, ignore_index = True)
    
    f_alpha = 0.05
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
                    resfolder = ACTOR + "_" + "_".join(INTERVENTION) + "_ttc_uninoise_03trg_0175risk_05alpha")
    
    
    
    
    
    fpcmci.run_filter()        
    fpcmci.filter_dependencies['r_v'] = []
            
    # list of selected features based on dependencies
    tmp_sel_features = fpcmci.get_selected_features()

    # shrink dataframe d and dependencies by the selector result
    fpcmci.shrink(tmp_sel_features)
        
    # selected links to check by the validator
    link_assumptions = fpcmci.get_link_assumptions()
            
    # causal model on selected links
    fpcmci.validator.data = fpcmci.data
    pcmci_result = fpcmci.validator.run(link_assumptions)
        
    # application of the validator result to the filter_dependencies field
    fpcmci.apply_validator_result(pcmci_result)
        
    fpcmci.result = fpcmci.get_selected_features()
    # shrink dataframe d and dependencies by the validator result
    fpcmci.shrink(fpcmci.result)
        
    # final causal model
    fpcmci.causal_model = fpcmci.validator.dependencies
    fpcmci.save_validator_res()
    
    colors = {r"r_{vh1}" : "orangered", 
              r"r_{vh2}" : "orangered", 
              r"r_{vt}" : "orangered", 
              r"r_v" : "orangered", 
              r"d_{rg}": "dodgerblue", 
              r"t_{rg}": "dodgerblue", 
              r"h_v": "dodgerblue", 
              r"risk": "dodgerblue", 
              r"d_{rh}" : "dodgerblue"}
    nodes_color = {'$' + key + '$': colors[key] for key in fpcmci.result}
    fpcmci.dag(label_type = LabelType.NoLabels, node_layout = 'circular', node_color = nodes_color)