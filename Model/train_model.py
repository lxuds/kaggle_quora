
"""
__file__

    train_model.py

__description__

    This file trains various models.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import sys
import csv
import os
import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from datetime import timedelta


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
## keras
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
'''
## cutomized module
from model_library_config import feat_folders, feat_names, param_spaces, int_feat
sys.path.append("../")
from param_config import config
#from ml_metrics import quadratic_weighted_logloss
from utils import *

from collections import Counter




global trial_counter
global log_handler


## libfm
libfm_exe = "../../libfm-1.40.windows/libfm.exe"

## rgf
call_exe = "../../rgf1.2/test/call_exe.pl"
rgf_exe = "../../rgf1.2/bin/rgf.exe"

output_path = "../../Output"

### global params
## you can use bagging to stabilize the predictions
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size=  1

ebc_hard_threshold = False
verbose_level = 1



def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
#    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
#        # It is a tie
#        return 0
    return top_two[0][0]


#### warpper for hyperopt for logging the training reslut
# adopted from
#
def hyperopt_wrapper(param, feat_folder, feat_name):
    global trial_counter
    global log_handler
    trial_counter += 1

    # convert integer feat
    for f in int_feat:
        if param.has_key(f):
            param[f] = int(param[f])

    print("------------------------------------------------------------")
    print "Trial %d" % trial_counter

    print("        Model")
    print("              %s" % feat_name)
    print("        Param")
    for k,v in sorted(param.items()):
        print("              %s: %s" % (k,v))
    print("        Result")
    print("                    Run      Fold      Bag      Logloss     Shape")

    ## evaluate performance
    logloss_cv_mean, logloss_cv_std = hyperopt_obj(param, feat_folder, feat_name, trial_counter)

    ## log
    var_to_log = [
        "%d" % trial_counter,
        "%.6f" % logloss_cv_mean, 
        "%.6f" % logloss_cv_std
    ]
    for k,v in sorted(param.items()):
        var_to_log.append("%s" % v)
    writer.writerow(var_to_log)
    log_handler.flush()

    return {'loss': logloss_cv_mean, 'attachments': {'std': logloss_cv_std}, 'status': STATUS_OK}
    

#### train CV and final model with a specified parameter setting
def hyperopt_obj(param, feat_folder, feat_name, trial_counter):

    logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    for run in range(1,config.n_runs+1):
        for fold in range(1,config.n_folds+1):
#    for run in [1]:
#        for fold in [1]:
            rng = np.random.RandomState(2015 + 1000 * run + 10 * fold)

            #### all the path
            path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
            save_path = "%s/Run%d/Fold%d" % (output_path, run, fold)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_model_path = "%s/Model/Run%d/Fold%d" % (output_path, run, fold)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)


            # feat
            feat_train_path = "%s/train.feat" % path
            feat_valid_path = "%s/valid.feat" % path
            # raw prediction path (rank)
            raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
            rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)

            # save trained model 
            cross_validation_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)


            ## load feat
            X_train, labels_train = load_svmlight_file(feat_train_path)
            X_valid, labels_valid = load_svmlight_file(feat_valid_path)
            if X_valid.shape[1] < X_train.shape[1]:
                X_valid = hstack([X_valid, np.zeros((X_valid.shape[0], X_train.shape[1]-X_valid.shape[1]))])
            elif X_valid.shape[1] > X_train.shape[1]:
                X_train = hstack([X_train, np.zeros((X_train.shape[0], X_valid.shape[1]-X_train.shape[1]))])
            X_train = X_train.tocsr()
            X_valid = X_valid.tocsr()
            Y_valid = labels_valid
            numTrain = X_train.shape[0]
            numValid = X_valid.shape[0]
            #print numTrain, numValid, Y_valid.shape
            ##############
            ## Training ##
            ##############
            ## you can use bagging to stabilize the predictions
            preds_bagging = np.zeros((numValid, bagging_size), dtype=float)
            for n in range(bagging_size):
                if bootstrap_replacement:
                    sampleSize = int(numTrain*bootstrap_ratio)
                    index_base = rng.randint(numTrain, size=sampleSize)
                    index_meta = [i for i in range(numTrain) if i not in index_base]
                else:
                    randnum = rng.uniform(size=numTrain)
                    index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
                    index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
                
                if param.has_key("booster"):
                    dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
                    dtrain_base = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])
                        
                    watchlist = []
                    if verbose_level >= 2:
                        watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
                    
                ## various models
                if param["task"] in ["regression"]:
                    ## classification  xgboost
#                    xgc = xgb.XGBClassifier(param)
                    clf = xgb.train(param, dtrain_base, param['num_round'], watchlist)
                    pred_proba = clf.predict(dvalid_base)
#                    pred = [int(round(x)) for x in pred_]

                elif param['task'] == "reg_skl_rf":
                    ## regression with sklearn random forest regressor
                    clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                               max_features=param['max_features'],
                                               n_jobs=param['n_jobs'],
                                               random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred = clf.predict(X_valid)
                    pred_proba = clf.predict_proba(X_valid)[:,1]
                    #clf_probs = clf.predict_proba(X_test)
                   

                elif param['task'] == "reg_skl_etr":
                    ## regression with sklearn extra trees regressor
                    clf = ExtraTreesClassifier(n_estimators=param['n_estimators'],
                                              max_features=param['max_features'],
                                              n_jobs=param['n_jobs'],
                                              random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]

                elif param['task'] == "reg_skl_gbm":
                    ## regression with sklearn gradient boosting regressor
                    clf = GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                                    max_features=param['max_features'],
                                                    learning_rate=param['learning_rate'],
                                                    max_depth=param['max_depth'],
                                                    subsample=param['subsample'],
                                                    random_state=param['random_state'])
                    clf.fit(X_train.toarray()[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid.toarray())[:,1]
                    

                elif param['task'] == "clf_skl_lr":
                    ## classification with sklearn logistic regression
                    clf = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            class_weight='auto', random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]

                elif param['task'] == "clf_skl_lr_l1":
                    ## classification with sklearn logistic regression with l1 penalty lasso
                    clf = LogisticRegression(penalty="l1", dual=True, tol=1e-5, solver ="liblinear",
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            class_weight='auto', random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]


                elif param['task'] == "reg_skl_svr":
                    ## regression with sklearn support vector regression
                    X_train, X_valid = X_train.toarray(), X_valid.toarray()
                    scaler = StandardScaler()
                    X_train[index_base] = scaler.fit_transform(X_train[index_base])
                    X_valid = scaler.transform(X_valid)
                    clf = SVC(C=param['C'], gamma=param['gamma'], 
                                            degree=param['degree'], kernel=param['kernel'], probability=True)
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]
                   
                elif param['task'] == "reg_skl_ridge":
                    ## regression with sklearn ridge regression
                    clf = RidgeClassifier(alpha=param["alpha"], normalize=True)
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict(X_valid)
                    

                ## this bagging iteration
                preds_bagging[:,n] = pred_proba
                pred_raw = np.mean(preds_bagging[:,:(n+1)], axis=1)
                pred_class_bagging = proba2class(pred_raw)              
                #logloss_valid = log_loss(Y_valid, pred_class_bagging)
                logloss_valid = log_loss(Y_valid, pred_proba)
                if (n+1) != bagging_size:
                    print("              {:>3}   {:>3}   {:>3}   {:>6}   {} x {}".format(
                                run, fold, n+1, np.round(logloss_valid,6), X_train.shape[0], X_train.shape[1]))
                else:
                    print("                    {:>3}       {:>3}      {:>3}    {:>8}  {} x {}".format(
                                run, fold, n+1, np.round(logloss_valid,6), X_train.shape[0], X_train.shape[1]))

                ### saving the training results on the training folds
                ### no bagging here
                ### saving the trained model
                with open(cross_validation_model_path, "wb") as f:
                    cPickle.dump(clf, f, -1)

            logloss_cv[run-1,fold-1] = logloss_valid
            ## save this prediction
            #print raw_pred_valid_path
            dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
            dfPred.to_csv(raw_pred_valid_path, index=False, header=True,
                         columns=["target", "prediction"])
            
            ## save this prediction
            dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_class_bagging})
            dfPred.to_csv(rank_pred_valid_path, index=False, header=True,
                         columns=["target", "prediction"])
            
    logloss_cv_mean = np.mean(logloss_cv)
    logloss_cv_std = np.std(logloss_cv)
    if verbose_level >= 1:
        print("              Mean: %.6f" % logloss_cv_mean)
        print("              Std: %.6f" % logloss_cv_std)

    
    ####################
    #### Retraining ####
    ####################
    #### all the path
    path = "%s/All" % (feat_folder)
    save_path = "%s/All" % output_path
    subm_path = "%s/Subm" % output_path
    save_model_path = "%s/Model/All" % output_path
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(subm_path):
        os.makedirs(subm_path)


    # feat
    feat_train_path = "%s/train.feat" % path
    #feat_test_path = "%s/test.feat" % path
    # raw prediction path (rank)
    #raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    #rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
    # submission path (relevance as in [1,2,3,4])
    #subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, logloss_cv_mean, logloss_cv_std)
    # save trained model
    all_training_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)

    #### load data
    ## load feat
    X_train, labels_train = load_svmlight_file(feat_train_path)
    #X_test, labels_test = load_svmlight_file(feat_test_path)
    #if X_test.shape[1] < X_train.shape[1]:
    #    X_test = hstack([X_test, np.zeros((X_test.shape[0], X_train.shape[1]-X_test.shape[1]))])
    #elif X_test.shape[1] > X_train.shape[1]:
    #    X_train = hstack([X_train, np.zeros((X_train.shape[0], X_test.shape[1]-X_train.shape[1]))])
    X_train = X_train.tocsr()
    #X_test = X_test.tocsr()
    numTrain = X_train.shape[0]
    #numTest = X_test.shape[0]
      
    ## bagging
    #preds_bagging = np.zeros((numTest, bagging_size), dtype=float)
    for n in range(bagging_size):
        if bootstrap_replacement:
            sampleSize = int(numTrain*bootstrap_ratio)
            #index_meta = rng.randint(numTrain, size=sampleSize)
            #index_base = [i for i in range(numTrain) if i not in index_meta]
            index_base = rng.randint(numTrain, size=sampleSize)
            index_meta = [i for i in range(numTrain) if i not in index_base]
        else:
            randnum = rng.uniform(size=numTrain)
            index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
            index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
 
        if param.has_key("booster"):
            #dtest = xgb.DMatrix(X_test, label=labels_test)
            dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])
                
            watchlist = []
            if verbose_level >= 2:
                watchlist  = [(dtrain, 'train')]
                    
        ## train
        if param["task"] in ["regression"]:
            clf = xgb.train(param, dtrain, param['num_round'], watchlist)
            #pred_proba = clf.predict(dtest)


        elif param['task'] == "reg_skl_rf":
            ## random forest regressor
            clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                       max_features=param['max_features'],
                                       n_jobs=param['n_jobs'],
                                       random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = clf.predict_proba(X_test)[:,1]

        elif param['task'] == "reg_skl_etr":
            ## extra trees regressor
            clf = ExtraTreesClassifier(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = clf.predict_proba(X_test)[:,1]

        elif param['task'] == "reg_skl_gbm":
            ## gradient boosting regressor
            clf = GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                            max_features=param['max_features'],
                                            learning_rate=param['learning_rate'],
                                            max_depth=param['max_depth'],
                                            subsample=param['subsample'],
                                            random_state=param['random_state'])
            clf.fit(X_train.toarray()[index_base], labels_train[index_base])
            #pred_proba = clf.predict_proba(X_test.toarray())[:,1]



        elif param['task'] == "clf_skl_lr":
            clf = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                    C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = lr.predict_proba(X_test)[:,1]
            '''
            w = np.asarray(range(1,numOfClass+1))
            pred = pred * w[np.newaxis,:]
            pred = np.sum(pred, axis=1)
            '''

        elif param['task'] == "clf_skl_lr_l1":
            clf = LogisticRegression(penalty="l1", dual=True, tol=1e-5, solver ="liblinear",
                                    C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = clf.predict_proba(X_test)[:,1]

        elif param['task'] == "reg_skl_svr":
            ## regression with sklearn support vector regression
            #X_train, X_test = X_train.toarray(), X_test.toarray()
            X_train = X_train.toarray()
            scaler = StandardScaler()
            X_train[index_base] = scaler.fit_transform(X_train[index_base])
            #X_test = scaler.transform(X_test)
            clf = SVC(C=param['C'], gamma=param['gamma'],
                                    degree=param['degree'], kernel=param['kernel'])
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = clf.predict_proba(X_test)[:,1]

        elif param['task'] == "reg_skl_ridge":
            clf = RidgeClassifier(alpha=param["alpha"], normalize=True)
            clf.fit(X_train[index_base], labels_train[index_base])
            #pred_proba = clf.predict(X_test)
        ## weighted averageing over different models
        #pred_test = pred_proba
        #preds_bagging[:,n] = pred_test
    #pred_raw = np.mean(preds_bagging, axis=1)
    #pred_class_bagging = proba2class(pred_raw)
    # save the retrained classifer result
    with open(all_training_model_path, "wb") as f:
        cPickle.dump(clf, f, -1)
    #no bagging used for retraining 
    '''
    ## write
    output = pd.DataFrame({"id": id_test, "prediction": pred_raw})    
    output.to_csv(raw_pred_test_path, index=False)

    ## write
    output = pd.DataFrame({"id": id_test, "prediction": pred_class_bagging})    
    output.to_csv(rank_pred_test_path, index=False)

    pred_score = pred_class_bagging
    output = pd.DataFrame({"id": id_test, "prediction": pred_score})    
    output.to_csv(subm_path, index=False)
    '''
        

    print ("----time elapsed----\n", str(timedelta(seconds=time.time() - start_time)))

    return logloss_cv_mean, logloss_cv_std



    
####################
## Model Buliding ##
####################

def check_model(models, feat_name):
#    print "check_model models:", models
#    print "feat_name", feat_name
    if models == "all":
        return True
    for model in models:
        if model in feat_name:
            return True
    return False


if __name__ == "__main__":
    start_time = time.time()
    
    specified_models = sys.argv[1:]
    print specified_models
    if len(specified_models) == 0:
        print("You have to specify which model to train.\n")
        print("Usage: python ./train_model_library_lsa.py model1 model2 model3 ...\n")
        print("Example: python ./train_model_library_lsa.py reg_skl_ridge reg_skl_lasso reg_skl_svr\n")
        print("See model_library_config_lsa.py for a list of available models (i.e., Model@model_name)")
        sys.exit()
    log_path = "%s/Log" % output_path
    print log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    for feat_name, feat_folder in zip(feat_names, feat_folders):
        if not check_model(specified_models, feat_name):
            continue
        param_space = param_spaces[feat_name]
        #"""
  
        log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
        log_handler = open( log_file, 'wb' )
        writer = csv.writer( log_handler )
        headers = [ 'trial_counter', 'logloss_mean', 'logloss_std' ]
        for k,v in sorted(param_space.items()):
            headers.append(k)
        writer.writerow( headers )
        log_handler.flush()
        
        print("************************************************************")
        print("Search for the best params")
        #global trial_counter
        trial_counter = 0
        trials = Trials()
        objective = lambda p: hyperopt_wrapper(p, feat_folder, feat_name)
        best_params = fmin(objective, param_space, algo=tpe.suggest,
                           trials=trials, max_evals=param_space["max_evals"])
        for f in int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k,v in best_params.items():
            print "        %s: %s" % (k,v)
        trial_loglosss = np.asarray(trials.losses(), dtype=float)
        best_logloss_mean = min(trial_loglosss)
        ind = np.where(trial_loglosss == best_logloss_mean)[0][0]
        best_logloss_std = trials.trial_attachments(trials.trials[ind])['std']
        print("Logloss stats")
        print("        Mean: %.6f\n        Std: %.6f" % (best_logloss_mean, best_logloss_std))
