
"""
__file__

    generate_ensemble_submission.py

__description__

    This file generates submission via ensemble selection.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import os
import sys
import numpy as np
import pandas as pd
from utils import *
from ensemble_selection import *
from model_library_config_tmp import feat_folders, feat_names


##
## config
model_folder = "../../Output"
subm_folder = "../../Output/Subm"
if not os.path.exists(subm_folder):
    os.makedirs(subm_folder)


## load test info
feat_folder = feat_folders[0]
info_test = pd.read_csv("%s/All/test.info" % feat_folder)
id_test = info_test["id"]
numTest = info_test.shape[0]

## reg
model_list = []
# try 5/10/50
id_sizes = 10*np.ones(len(feat_names), dtype=int)
for feat_name,id_size in zip(feat_names, id_sizes):
    ## get the top 10 model ids
    log_file = "%s/Log/%s_hyperopt.log" % (model_folder, feat_name)
    try:
        dfLog = pd.read_csv(log_file)
        #print dfLog.columns
        # dfLog.sort("logloss_mean", ascending=True, inplace=True)
        dfLog.sort_values(by="logloss_mean", ascending=True, inplace=True)
        ind = np.min([id_size, dfLog.shape[0]])
        ids = dfLog.iloc[:ind]["trial_counter"]
        #print dfLog[:ind]
        model_list += ["%s_[Id@%d]" % (feat_name, id) for id in ids]
    except:
        pass

  

bagging_size = 10
bagging_fraction = 1.0
prunning_fraction = 1.
bagging_replacement = True
init_top_k = 5

subm_prefix = "%s/test.pred.[ensemble_selection]_[Solution]" % (subm_folder)
best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight = \
    ensembleSelection(feat_folder, model_folder, model_list,  subm_prefix=subm_prefix, \
        hypteropt_max_evals=1, w_min=-1, w_max=1, bagging_replacement=bagging_replacement, bagging_fraction=bagging_fraction, \
        bagging_size=bagging_size, init_top_k=init_top_k, prunning_fraction=prunning_fraction)







