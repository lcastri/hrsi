from tigramite.independence_tests import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
from time import time
from datetime import timedelta
import pandas as pd

DATA_PATH = r'/home/lucacastri/Git/darko_ws/src/hrsi/tiago_data_handler/data'
NUM_DATASET = 6
ACTOR = "sariah"
# INTERVENTION = "noaction"
# INTERVENTION = "decrease"
INTERVENTION = "increase"
FILE_EXT = ".csv"




if __name__ == '__main__':
    all_files = [DATA_PATH + "/" + ACTOR + "_" + INTERVENTION + "_" + str(i) + "_causal" + FILE_EXT for i in range(NUM_DATASET)]

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col = [0], header = 0)
        li.append(df)

    frame = pd.concat(li, axis = 0, ignore_index = True)
    
    
    
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = Data(frame)
    start = time()
    fpcmci = FPCMCI(df, 
                alpha = alpha, 
                min_lag = min_lag, 
                max_lag = max_lag, 
                sel_method = TE(TEestimator.Gaussian), 
                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                verbosity = CPLevel.DEBUG,
                neglect_only_autodep = True,
                resfolder = ACTOR + "_" + INTERVENTION)
    
    fpcmci_res = fpcmci.run()
    elapsed_FPCMCI = time() - start
    print(str(timedelta(seconds = elapsed_FPCMCI)))
    fpcmci.dag(label_type = LabelType.NoLabels, node_layout = 'dot')