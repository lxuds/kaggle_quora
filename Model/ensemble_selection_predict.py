
"""
__file__

    ensemble_selection_predict.py

__description__

    This file reads in the model lists and model weights generated from ensemble selection, and make predictions on testing sample.

__author__

    Lei Xu < leixuast@gmail.com >

"""


import csv
import sys
import numpy as np
import pandas as pd
from utils import getScore, getTestScore, proba2class
#from ml_metrics import quadratic_weighted_logloss
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
sys.path.append("../")
from param_config import config
from sklearn.metrics import log_loss

########################
## Ensemble Selection ##
########################
def ensembleSelectionPrediction(model_folder, best_bagged_model_list, best_bagged_model_weight, subset):
    bagging_size = len(best_bagged_model_list)
    for bagging_iter in range(bagging_size):
        w_ens = 0
        iter = 0
        for model,w in zip(best_bagged_model_list[bagging_iter], best_bagged_model_weight[bagging_iter]):
            iter += 1
            pred_file = "%s/All/test.%s.pred.%s.csv" % (model_folder, subset, model)
            print pred_file
            this_p_valid = pd.read_csv(pred_file, dtype=float)["prediction"].values
            this_w = w
            if iter == 1:
                p_ens_valid = np.zeros((this_p_valid.shape[0]),dtype=float)
                id_test = pd.read_csv(pred_file, dtype=float)["id"].values
                id_test = np.asarray(id_test, dtype=int)
            p_ens_valid = (w_ens * p_ens_valid + this_w * this_p_valid) / (w_ens + this_w)
            w_ens += this_w
        if bagging_iter == 0:
            p_ens_valid_bag = p_ens_valid
        else:
            p_ens_valid_bag = (bagging_iter * p_ens_valid_bag + p_ens_valid) / (bagging_iter+1.)
    p_ens_score = proba2class(p_ens_valid_bag)
    ##
    output = pd.DataFrame({"id": id_test, "prediction": p_ens_score})
    return output



if __name__ == "__main__":

    exec("Ntest ="+ sys.argv[1])
    print Ntest
    model_folder = "../../Output"
    subm_folder = "../../Output/Subm"
    
    bagging_size = 10
    bagging_fraction = 1.0
    prunning_fraction = 1.
    bagging_replacement = True
    init_top_k = 5
    
    
    save_bagging_path = "%s/bagging" % model_folder
    
    save_bagging_log_name = "%s/bagging_[InitTopK%d]_[BaggingFraction%s].csv"\
                                % (save_bagging_path, init_top_k, bagging_fraction)
    bagging_log = pd.read_csv(save_bagging_log_name)
    bagging_file_names = bagging_log["bagging_file_name"].values
    bagging_best_logloss = bagging_log["bagged_best_logloss_mean"].values
    bagging_best_logloss_std = bagging_log[ "bagged_best_logloss_std"].values
    
    best_bagged_model_list = [[]]*bagging_size
    best_bagged_model_weight = [[]]*bagging_size
    # re-generate greedy ensemble model list and weights
    for bagging_iter in range(bagging_size):
        bagging_file_name = bagging_file_names[bagging_iter]
        this_best_bagging = pd.read_csv(bagging_file_name)
        this_best_bagged_model_list = this_best_bagging["best_bagged_model"]
        this_best_bagged_model_weight = this_best_bagging["best_bagged_model_weight"]
        best_bagged_model_list[bagging_iter] = this_best_bagged_model_list 
        best_bagged_model_weight[bagging_iter] = this_best_bagged_model_weight 

    for i in Ntest:
        print "For Testing set the %sth subset" % str(i)
        for bagging_iter in range(bagging_size):
            this_bagging_best_logloss = bagging_best_logloss[bagging_iter]
            this_bagging_best_logloss_std = bagging_best_logloss_std[bagging_iter]
            ## save the current prediction

            subm_prefix = "%s/test.%s.pred.[ensemble_selection]_[Solution]" % (subm_folder, i)
            output = ensembleSelectionPrediction(model_folder, best_bagged_model_list[:(bagging_iter+1)], best_bagged_model_weight[:(bagging_iter+1)], i)
            sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cdf.csv" \
                    % (subm_prefix, init_top_k, bagging_iter+1, bagging_fraction, bagging_best_logloss[bagging_iter], bagging_best_logloss_std[bagging_iter])
            output.to_csv(sub_file, index=False)
            print sub_file
        print "===================end of bagging iteration====================="
     
     

