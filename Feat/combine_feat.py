
"""
__file__

    combine_feat.py

__description__

    This file provides modules for combining features and save them in svmlight format.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import os
import sys
import cPickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file
sys.path.append("../")
from param_config import config
import time
from datetime import timedelta



#### adopted from @Ben Hamner's Python Benchmark code
## https://www.kaggle.com/benhamner/crowdflower-search-relevance/python-benchmark
def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)


#### function to combine features
def combine_feat(feat_names, feat_path_name, mode, subset_id=None):
    
    print("==================================================")
    print("Combine features...")
    
    start_time = time.time()

    if mode == 'Training':
    
        ######################
        ## Cross-validation ##
        ######################
        print("For cross-validation...")
        ## for each run and fold

        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = cPickle.load(f)


        for run in range(1,config.n_runs+1):
            for fold in range(1,config.n_folds+1):
                print("Run: %d, Fold: %d" % (run, fold))
                path = "%s/Run%d/Fold%d" % (config.feat_folder, run, fold)
                info_path = "%s/Run%d/Fold%d" % (config.feat_folder, run, fold)
                save_path = "%s/%s/Run%d/Fold%d" % (config.feat_folder, feat_path_name, run, fold)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for i,(feat_name,transformer) in enumerate(feat_names):

                    ## load train feat
                    feat_train_file = "%s/train.%s.feat.pkl" % (path, feat_name)
                    with open(feat_train_file, "rb") as f:
                        x_train = cPickle.load(f)
                    if len(x_train.shape) == 1:
                        #print x_train.shape
                        x_train.shape = (x_train.shape[0], 1)

                    ## load valid feat
                    feat_valid_file = "%s/valid.%s.feat.pkl" % (path, feat_name)
                    with open(feat_valid_file, "rb") as f:
                        x_valid = cPickle.load(f)
                        #print x_valid.shape
                    if len(x_valid.shape) == 1:
                        x_valid.shape = (x_valid.shape[0], 1)

                    ## align feat dim
                    dim_diff = abs(x_train.shape[1] - x_valid.shape[1])
                    print dim_diff, feat_name, x_train.shape[1], x_valid.shape[1], x_train.shape[0], x_valid.shape[0], run, fold
                    if dim_diff > 0:
                       print "training and valid sets have different feature diamensions:", x_train.shape[1] - x_valid.shape[1]
                       print dim_diff, feat_name, x_train.shape[1], x_valid.shape[1], x_train.shape[0], x_valid.shape[0]
                       return
                    if x_valid.shape[1] < x_train.shape[1]:
                        x_valid = hstack([x_valid, np.zeros((x_valid.shape[0], dim_diff))]).tocsr()
                    elif x_valid.shape[1] > x_train.shape[1]:
                        x_train = hstack([x_train, np.zeros((x_train.shape[0], dim_diff))]).tocsr()

                    ## apply transformation
                    x_train = transformer.fit_transform(x_train)
                    x_valid = transformer.transform(x_valid)

                    ## stack feat
                    if i == 0:
                        X_train, X_valid = x_train, x_valid
                    else:
                        try:
                            X_train, X_valid = hstack([X_train, x_train]), hstack([X_valid, x_valid])
                        except:
                            X_train, X_valid = np.hstack([X_train, x_train]), np.hstack([X_valid, x_valid])

                    #print("Combine {:>2}/{:>2} feat: {} ({}D)".format(i+1, len(feat_names), feat_name, x_train.shape[1]))
                print "Feat dim: {}D".format(X_train.shape[1])

                ## load label

                # train
                info_train = pd.read_csv("%s/train.info" % (info_path))
                # print info_train.columns
                ## change it to zero-based for multi-classification in xgboost
                Y_train = info_train["is_duplicate"] 
                # valid               
                info_valid = pd.read_csv("%s/valid.info" % (info_path))
                Y_valid = info_valid["is_duplicate"] 
                
                ## dump feat
                dump_svmlight_file(X_train, Y_train, "%s/train.feat" % (save_path))
                dump_svmlight_file(X_valid, Y_valid, "%s/valid.feat" % (save_path))
                
    print "All done"
    ##########################
    ## Training and Testing ##
    ##########################

    if mode == 'Testing':
        for j in subset_id:                
            print "For Testing set No. %s subset" % str(j)
            path = "%s/All" % (config.feat_folder)
            save_path = "%s/%s/All" % (config.feat_folder, feat_path_name)
            info_path = "%s/All" % (config.feat_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            ## load testing data
            for i,(feat_name,transformer) in enumerate(feat_names):
                #print feat_name
                ## load train feat
                feat_train_file = "%s/train.%s.feat.pkl" % (path, feat_name)
                with open(feat_train_file, "rb") as f:
                    x_train = cPickle.load(f)
                #print feat_train_file
                #print "x_train.shape:", x_train.shape
                if len(x_train.shape) == 1:
                    x_train.shape = (x_train.shape[0], 1)
         
                ## load test feat
                feat_test_file = "%s/test.%s.%s.feat.pkl" % (path, str(j), feat_name)
                with open(feat_test_file, "rb") as f:
                    x_test = cPickle.load(f)
                #print "x_test.shape:", x_test.shape
                if len(x_test.shape) == 1:
                    x_test.shape = (x_test.shape[0], 1)
                
                ## align feat dim
                dim_diff = abs(x_train.shape[1] - x_test.shape[1])
                #print dim_diff, feat_name, x_train.shape[1], x_test.shape[1], x_train.shape[0], x_test.shape[0]
                if dim_diff > 0:
                    print "training and testing sets have different feature diamensions:", x_train.shape[1] - x_test.shape[1]
                    print ("%s subset" %(j))
                    print dim_diff, feat_name, x_train.shape[1], x_test.shape[1], x_train.shape[0], x_test.shape[0]
                    return 
                '''
                if x_test.shape[1] < x_train.shape[1]:
                    x_test = hstack([x_test, np.zeros((x_test.shape[0], dim_diff))]).tocsr()
                elif x_test.shape[1] > x_train.shape[1]:
                    x_train = hstack([x_train, np.zeros((x_train.shape[0], dim_diff))]).tocsr()
                '''
                ## apply transformation
                x_train = transformer.fit_transform(x_train)
                x_test = transformer.transform(x_test)
                ## stack feature for training sets only once
                if j == 1:
                    if i == 0:
                        X_train = x_train
                    else:
                        try: 
                            X_train = hstack([X_train, x_train])
                        except:
                            X_train = np.hstack([X_train, x_train]) 
               
                ## stack feat
                if i == 0:
                    X_test = x_test
                else:
                    try:
                        X_test = hstack([X_test, x_test])
                    except:
                        X_test = np.hstack([X_test, x_test])
         
                print("Combine {:>2}/{:>2} feat: {} ({}D)".format(i+1, len(feat_names), feat_name, x_train.shape[1]))
                print "Feat dim: {}D".format(X_train.shape[1])
                
            ## load label and dump feat
            # train
            if j ==1:
                info_train = pd.read_csv("%s/train.info" % (info_path))
                Y_train = info_train["is_duplicate"]  
                dump_svmlight_file(X_train, Y_train, "%s/train.feat" % (save_path))
            # test 
            info_test = pd.read_csv("%s/test.%s.info" % (info_path,j))
            Y_test = info_test["is_duplicate"] 
            dump_svmlight_file(X_test, Y_test, "%s/test.%s.feat" % (save_path,j))
            print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
            print ("All done")
            print ("===============================================================")
         
            print (" \n\n\n ")
       
       
