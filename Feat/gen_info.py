
"""
__file__

    gen_info.py

__description__

    This file generates the following info for each run and fold, and for the entire training and testing set.

        1. training and validation/testing data

        2. sample weight

        3. cdf of the median_relevance
        
        4. the group info for pairwise ranking in XGBoost

__author__

    Lei Xu < leixuast@gmail.com >

"""

import os
import sys
import cPickle
import numpy as np
import pandas as pd
sys.path.append("../")
from param_config import config

if __name__ == "__main__":
#def gen_info():
    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)
    print dfTrain.columns

    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
        skf = cPickle.load(f)
        
    #######################
    ## Generate Features ##
    #######################
    print("Generate info...")
    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (trainInd, validInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
            if not os.path.exists(path):
                os.makedirs(path)

            #############################    
            ## get and dump group info ##
            #############################
            np.savetxt("%s/train.feat.group" % path, [len(trainInd)], fmt="%d")
            np.savetxt("%s/valid.feat.group" % path, [len(validInd)], fmt="%d")
            
               
            #############################
            ## dump all the other info ##
            #############################
            #print dfTrain.iloc[trainInd]
            dfTrain["is_duplicate"].iloc[trainInd].to_csv("%s/train.info" % path, index=False, header=True,encoding='utf-8')
            dfTrain["is_duplicate"].iloc[validInd].to_csv("%s/valid.info" % path, index=False, header=True,encoding='utf-8')
    print("Done.")

    print("For training and testing...")
    path = "%s/All" % (config.feat_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    ## weight
   
    ## group
    np.savetxt("%s/All/train.feat.group" % (config.feat_folder), [dfTrain.shape[0]], fmt="%d")
    ## info        
    dfTrain["is_duplicate"].to_csv("%s/All/train.info" % (config.feat_folder), index=False, header=True,encoding='utf-8')

    print "Done info extraction for training data"
    ###############
    ## Load Data ##
    ###############
    ## load testing data
    for i in range(config.test_subset_number):
        print "For Testing set No. %s subset" % str(i)
        with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "rb") as f:
            dfTest = cPickle.load(f) 
            dfTest[["is_duplicate","id"]].to_csv("%s/All/test.%s.info" % (config.feat_folder, i), index=False, header=True,encoding='utf-8')
            np.savetxt("%s/All/test.feat.group" % (config.feat_folder), [dfTest.shape[0]], fmt="%d")

    print("All Done.")
