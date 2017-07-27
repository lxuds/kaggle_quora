
"""
__file__

    single_model_predict_test.py

__description__

    This file loads training results of models with various hyper-parameters, and makes predictions on testing sets

__author__

    Lei Xu < leixuast@gmail.com >

"""

import time
import sys
import csv
import os
import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import RidgeClassifier, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.models import Sequential
## cutomized module
from model_library_config import feat_folders, feat_names, param_spaces, int_feat
sys.path.append("../")
from param_config import config
from utils import proba2class
from collections import Counter
from datetime import timedelta



def gen_prediction(param, feat_folder, feat_name, trial_counters, logloss_cv_means, logloss_cv_stds, subset):

    ### all the path
    path = "%s/All" % (feat_folder)
    feat_test_path = "%s/test.%s.feat" % (path, subset)
    subm_path = "%s/Subm" % output_path
    save_path = "%s/All" % output_path
    info_path = "%s/All" % (config.feat_folder)
    info_test = pd.read_csv("%s/test.%s.info" % (info_path,subset))
    #print info_test.columns
    id_test = info_test["id"]
    ## load feat
    X_test, labels_test = load_svmlight_file(feat_test_path)
    numTest = X_test.shape[0]
    print numTest
    X_test = X_test.tocsr()


    for trial_counter, logloss_cv_mean, logloss_cv_std in zip (trial_counters, logloss_cv_means, logloss_cv_stds): 
        save_retraining_path = "%s/test.%s.pred.%s_[Id@%d].csv" \
                                % (save_path, subset, feat_name, trial_counter)
        save_training_foldsmean_path = "%s/test.%s.foldsmean.pred.%s_[Id@%d].csv"\
                                % (save_path, subset, feat_name, trial_counter)
        subm_retraining_class_path = "%s/test.%s.pred_class.%s_[Id@%d].csv" \
                                % (subm_path, subset, feat_name, trial_counter)
        subm_training_foldsmean_class_path = "%s/test.%s.foldsmean.pred_class.%s_[Id@%d].csv"\
                                % (subm_path, subset, feat_name, trial_counter)

        pred_folds = np.zeros((config.n_runs, config.n_folds, numTest), dtype=float)
        for run in range(1,config.n_runs+1):
            for fold in range(1,config.n_folds+1):
                rng = np.random.RandomState(2015 + 1000 * run + 10 * fold)
     
                ## load model
                save_model_path = "%s/Model/Run%d/Fold%d" % (output_path, run, fold)
                cross_validation_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)
                with open(cross_validation_model_path, 'rb') as f:
                    clf = cPickle.load(f)
                   
                if "booster" in param:
                    dtest = xgb.DMatrix(X_test, label=labels_test)
     
                ## various models
                if param["task"] in ["regression"]:
                    ## classification  xgboost
                    pred_proba = clf.predict(dtest)
     
                elif param['task'] in ["reg_skl_rf", "reg_skl_etr", "reg_skl_gbm", "clf_skl_lr_l1",
                                        "clf_skl_lr_l1", "reg_skl_svr"]:
                    pred_proba = clf.predict_proba(X_test)[:,1]
                   
                elif param['task'] == "reg_skl_ridge":
                    pred_proba = clf.predict(X_test)
                    
                #pred_class = proba2class(pred_proba)              
                pred_folds[run-1,fold-1,:] = pred_proba
                
        pred_folds_mean = np.mean(np.mean(pred_folds, axis=0), axis=0)
        pred_folds_mean_class =  proba2class(pred_folds_mean)

        ## write
        output = pd.DataFrame({"id": id_test, "prediction": pred_folds_mean})    
        output.to_csv(save_training_foldsmean_path, index=False)

        output_class = pd.DataFrame({"id": id_test, "prediction": pred_folds_mean_class})
        output_class.to_csv(subm_training_foldsmean_class_path, index=False)

    
        #### Using retraining on the whole training set to make predictions on testing sets 
        save_model_path = "%s/Model/All" % output_path
        all_training_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)
        ## load model
        with open(all_training_model_path, 'rb') as f:
            clf = cPickle.load(f)       
        ## various models
        if param["task"] in ["regression"]:
            ## classification  xgboost
            pred_proba = clf.predict(dtest)
        elif param['task'] in ["reg_skl_rf", "reg_skl_etr", "reg_skl_gbm", "clf_skl_lr_l1",
                                "clf_skl_lr_l1", "reg_skl_svr"]:
            pred_proba = clf.predict_proba(X_test)[:,1]
        elif param['task'] == "reg_skl_ridge":
            pred_proba = clf.predict(X_test)
     
        pred_all_class = proba2class(pred_proba)
        pred_all = pred_proba
        ## write
        output = pd.DataFrame({"id": id_test, "prediction": pred_all})    
        output.to_csv(save_retraining_path, index=False)
        output_class = pd.DataFrame({"id": id_test, "prediction": pred_all_class})
        output_class.to_csv(subm_retraining_class_path, index=False) 

        print ("----time elapsed----\n", str(timedelta(seconds=time.time() - start_time)))
    
    return 
    

    
        
####################
## Model Buliding ##
####################
if __name__ == "__main__":


    start_time = time.time()
    
    specified_models = sys.argv[1]
    exec("Ntest ="+ sys.argv[2])
    print specified_models
    if len(specified_models) == 0:
        print("You have to specify which model to predict.\n")
        sys.exit()

    output_path = "../../Output"
    log_path = "%s/Log" % output_path
    #print log_path
    for feat_name, feat_folder in zip(feat_names, feat_folders):
        #print "specified_models:", specified_models
        #print "feat_name:", feat_name
        #if not check_model(specified_models, feat_name):
        #    continue
        if specified_models == feat_name: 
    
            print feat_name
            param_space = param_spaces[feat_name]
            log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
            dfLog = pd.read_csv(log_file)
            for i in Ntest:
               print "for No %s subset" % i
               gen_prediction(param_space, feat_folder, feat_name, dfLog["trial_counter"], dfLog["logloss_mean"], dfLog["logloss_std"], i)

