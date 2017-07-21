
"""
__file__

    ensemble_selection.py

__description__

    This file contains ensemble selection module.

__author__

    Lei Xu < leixuast@gmail.com >

"""


import csv
import sys
import numpy as np
import pandas as pd
import os
from utils import getScore, getTestScore, proba2class
#from ml_metrics import quadratic_weighted_logloss
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
sys.path.append("../")
from param_config import config
from sklearn.metrics import log_loss

########################
## Ensemble Selection ##
########################
def ensembleSelectionObj(param, p1_list, weight1, p2_list, true_label_list,  numValidMatrix):

    weight2 = param['weight2']
    logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    for run in range(config.n_runs):
        for fold in range(config.n_folds):
            numValid = numValidMatrix[run][fold]
            p1 = p1_list[run,fold,:numValid]
            p2 = p2_list[run,fold,:numValid]
            true_label = true_label_list[run,fold,:numValid]
            p_ens = (weight1 * p1 + weight2 * p2) / (weight1 + weight2)
            pred_class = proba2class(p_ens)
            logloss_cv[run][fold] = log_loss(true_label, p_ens)
    logloss_cv_mean = np.mean(logloss_cv)
    return {'loss': logloss_cv_mean, 'status': STATUS_OK}



#############

def ensembleSelection(feat_folder, model_folder, model_list, subm_prefix,
                 hypteropt_max_evals=10, w_min=-1., w_max=1.,
                  bagging_replacement=False, bagging_fraction=0.5, bagging_size=10, init_top_k=5, prunning_fraction=0.2):
    ## load all the prediction
    maxNumValid = 140000
    pred_list_valid = np.zeros((len(model_list), config.n_runs, config.n_folds, maxNumValid), dtype=float)
    Y_list_valid = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)
    numValidMatrix = np.zeros((config.n_runs, config.n_folds), dtype=int)
    p_ens_list_valid = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)

    numTest = 100000 
    
    ## model to idx
    model2idx = dict()
    logloss_list = dict()
    for i,model in enumerate(model_list): 
        model2idx[model] = i
        logloss_list[model] = 0
    print("============================================================")
    print("Load model...")
    for model in model_list:
        model_id = model2idx[model]
        print("model: %s" % model)
        logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                path = "%s/Run%d/Fold%d" % (model_folder, run+1, fold+1)
                pred_file = "%s/valid.raw.pred.%s.csv" % (path, model)
                this_p_valid = pd.read_csv(pred_file, dtype=float)
                numValidMatrix[run][fold] = this_p_valid.shape[0]           
                pred_list_valid[model_id,run,fold,:numValidMatrix[run][fold]] = this_p_valid["prediction"].values
                Y_list_valid[run,fold,:numValidMatrix[run][fold]] = this_p_valid["target"].values
                score = log_loss(this_p_valid["target"].values, this_p_valid["prediction"].values)
                logloss_cv[run][fold] = score
        print("logloss: %.6f" % np.mean(logloss_cv))
        logloss_list[model] = np.mean(logloss_cv)
    sorted_models = sorted(logloss_list.items(), key=lambda x: x[1]) #[::-1] best the models are on the top
    #print logloss_list.items
    #return       
    # greedy ensemble
    print("============================================================")
    print("Perform ensemble selection...")
    best_bagged_model_list = [[]]*bagging_size
    best_bagged_model_weight = [[]]*bagging_size
    num_model = len(model_list)

    bagged_best_logloss_mean = []
    bagged_best_logloss_std = []
    bagged_model_list_file = []

    #print bagging_size
    for bagging_iter in range(bagging_size):
        print(" \n")
        print("===============================")
        print("New bagging iteration...")
        print("===============================")
        rng = np.random.RandomState(2015 + 100 * bagging_iter)
        if bagging_replacement:
            sampleSize = int(num_model*bagging_fraction)
#            print sampleSize, num_model
            index_base = rng.randint(num_model, size=sampleSize)
        else:
            randnum = rng.uniform(size=num_model)
            index_base = [i for i in range(num_model) if randnum[i] < bagging_fraction]
        this_sorted_models = [sorted_models[i] for i in sorted(index_base)]

        #print this_model_list
        best_model_list = []
        best_model_weight = []
        best_logloss = 0
        best_model = None
        p_ens_list_valid_tmp = np.zeros((config.n_runs, config.n_folds, maxNumValid), dtype=float)
        #### initialization
        w_ens, this_w = 0, 1.0
        if init_top_k > 0:
            cnt = 0
            logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
            for model,logloss in this_sorted_models:
                if cnt >= init_top_k:
                    continue
                print("add to the ensembles the following model")
                print("model: %s" % model)
                print("logloss: %.6f" % logloss)
                this_p_list_valid = pred_list_valid[model2idx[model]]
                for run in range(config.n_runs):
                    for fold in range(config.n_folds):
                        numValid = numValidMatrix[run][fold]
                        if cnt == 0:
                            this_w = 1.0
                        else:
                            pass
                        p_ens_list_valid_tmp[run,fold,:numValid] = (w_ens * p_ens_list_valid_tmp[run,fold,:numValid] + this_w * this_p_list_valid[run,fold,:numValid]) / (w_ens + this_w)
                        if cnt == init_top_k - 1:
                            true_label = Y_list_valid[run,fold,:numValid]
                            pred_class = proba2class(p_ens_list_valid_tmp[run,fold,:numValid])
                            logloss = log_loss(true_label, p_ens_list_valid_tmp[run,fold,:numValid])
                            logloss_cv[run][fold] = logloss
                best_model_list.append(model)
                best_model_weight.append(this_w)
                best_logloss = np.mean(logloss_cv)
                w_ens += this_w
                cnt += 1
            print("Init logloss: %.6f (%.6f)" % (np.mean(logloss_cv), np.std(logloss_cv)))
            print("-----------------")


        #### ensemble selection with replacement
        iter = 0 
        d = {}
        k = best_model_list
        v = best_model_weight
        item = list(zip(k, v))
        for (x,y) in item:
            if x in d:
               d[x] = d[x] + y #or whatever your function needs to be to combine them
            else:
               d[x] = y
        d = {k: v for k, v in d.items() if v > 0}
        best_model_list_prunning = d.keys()
        best_model_weight_prunning = d.values()
        print 'model weight list', d.values()
        while True:
            iter += 1
            for ii, (model,_) in enumerate(this_sorted_models):
                this_p_list_valid = pred_list_valid[model2idx[model]]

                ## hyperopt for the best weight
                trials = Trials()
                param_space = {
                    'weight2': hp.uniform('weight2', w_min, w_max)
                }
                obj = lambda param: ensembleSelectionObj(param, p_ens_list_valid_tmp, 1., this_p_list_valid, Y_list_valid,  numValidMatrix)
                best_params = fmin(obj,
                                   param_space, algo=tpe.suggest,
                                   trials=trials, max_evals=hypteropt_max_evals)
                this_w = best_params['weight2']
                this_w *= w_ens
                # all the current prediction to the ensemble
                logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
                for run in range(config.n_runs):
                    for fold in range(config.n_folds):
                        numValid = numValidMatrix[run][fold]
                        p1 = p_ens_list_valid_tmp[run,fold,:numValid]
                        p2 = this_p_list_valid[run,fold,:numValid]
                        true_label = Y_list_valid[run,fold,:numValid]
                        p_ens = (w_ens * p1 + this_w * p2) / (w_ens + this_w)
                        pred_class = proba2class(p_ens)
                        score = log_loss(true_label, p_ens)
                        logloss_cv[run][fold] = score 

                if (np.mean(logloss_cv) < best_logloss):
                   if model in d: 
                      weight_tmp = d[model] + this_w
                   else:
                      weight_tmp = this_w
                   if weight_tmp >0:
                      print '  '
                      print '     model:', ii 
                      print '     this model:', model
                      print '     this model weight:', this_w
                      print '     this model total weight:', weight_tmp
                      #d[model] = weight_tmp
                      #print 'model weight list', d.values()
                      best_logloss, best_model, best_weight = np.mean(logloss_cv), model, this_w

            if best_model == None:
                break
            print("Iter: %d" % iter)
            print("    model: %s" % best_model)
            print("    weight: %s" % best_weight)
            print("    logloss: %.6f" % best_logloss)

            best_model_list.append(best_model)
            best_model_weight.append(best_weight)
            print '    before adding this model, model weight list', d.values()
            if best_model in d:
               d[best_model] = d[best_model] + best_weight
            else:
               d[best_model] = best_weight
            print '    after adding this model, model weight list', d.values()
            print '  '
            print '----'
            # valid
            this_p_list_valid = pred_list_valid[model2idx[best_model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    numValid = numValidMatrix[run][fold]
                    p_ens_list_valid_tmp[run,fold,:numValid] = (w_ens * p_ens_list_valid_tmp[run,fold,:numValid] + best_weight * this_p_list_valid[run,fold,:numValid]) / (w_ens + best_weight)
            best_model = None
            w_ens += best_weight   # end of while loop

        # use the ensemble selection best models to calculate p_ens_list_valid and logloss: here is a cumulative result from previous iterations. average each iteration weight is 1  
        logloss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        cutoff = np.zeros((3), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                numValid = numValidMatrix[run][fold]
                true_label = Y_list_valid[run,fold,:numValid]
                p_ens_list_valid[run,fold,:numValid] = (bagging_iter * p_ens_list_valid[run,fold,:numValid] + p_ens_list_valid_tmp[run,fold,:numValid]) / (bagging_iter+1.)
                pred_class = proba2class(p_ens_list_valid[run,fold,:numValid])
                logloss_cv[run][fold] = log_loss(true_label, p_ens_list_valid[run,fold,:numValid])
        print( "Bag %d, logloss: %.6f (%.6f)" % (bagging_iter+1, np.mean(logloss_cv), np.std(logloss_cv)) )
        best_logloss_mean = np.mean(logloss_cv)
        best_logloss_std = np.std(logloss_cv)
        best_bagged_model_list[bagging_iter] = best_model_list
        best_bagged_model_weight[bagging_iter] = best_model_weight

        save_bagging_path = "%s/bagging" % model_folder 
        if not os.path.exists(save_bagging_path):
            os.makedirs(save_bagging_path)
        save_this_bagging_file_name = "%s/bagging_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s].csv" \
                               % (save_bagging_path, init_top_k, bagging_iter+1, bagging_fraction)
        this_best_bagging = pd.DataFrame({"best_bagged_model": best_model_list, "best_bagged_model_weight": best_model_weight})        
        this_best_bagging.to_csv(save_this_bagging_file_name, index=False)

        bagged_best_logloss_mean = bagged_best_logloss_mean + [best_logloss_mean]
        bagged_best_logloss_std = bagged_best_logloss_std + [best_logloss_std]
        bagged_model_list_file = bagged_model_list_file + [save_this_bagging_file_name]


        print "===================end of bagging iteration====================="
    save_bagging_log_name = "%s/bagging_[InitTopK%d]_[BaggingFraction%s].csv"\
                            % (save_bagging_path, init_top_k, bagging_fraction)
    bagging_result_file_list = pd.DataFrame({"bagging_file_name":bagged_model_list_file, \
                             "bagged_best_logloss_mean":bagged_best_logloss_mean, "bagged_best_logloss_std":bagged_best_logloss_std}) 
    bagging_result_file_list.to_csv(save_bagging_log_name, index=False)
    return best_logloss_mean, best_logloss_std, best_bagged_model_list, best_bagged_model_weight
