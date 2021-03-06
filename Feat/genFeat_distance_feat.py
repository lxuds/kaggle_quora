
"""
__file__

    genFeat_distance_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. jaccard coefficient/dice distance between 
            - just plain jaccard coefficient/dice distance between question1 and question2
            - compute for unigram/bigram/trigram

__author__

    Lei Xu < leixuast@gmail.com >

"""

import re
import sys
import ngram
import cPickle
import numpy as np
import pandas as pd
import time
from copy import copy
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, get_sample_indices_by_relevance, dump_feat_name
sys.path.append("../")
from param_config import config
from functools import partial
from datetime import timedelta

stats_feat_flag = False


#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

#### pairwise distance
def pairwise_jaccard_coef(A, B):
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i,j] = JaccardCoef(A[i], B[j])
    return coef
    
def pairwise_dice_dist(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i,j] = DiceDist(A[i], B[j])
    return d

def pairwise_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = pairwise_jaccard_coef(A, B)
    elif dist == "dice_dist":
        d = pairwise_dice_dist(A, B)
    return d


######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
transform = config.count_feat_transform
def preprocess_data(line, token_pattern=token_pattern,
                     exclude_stopword=config.cooccurrence_word_exclude_stopword,
                     encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed


#####################################
## Extract basic distance features ##
#####################################
def extract_basic_distance_feat(df):
    ## unigram
    
    ## unigram
    print "generate ngrams"
    join_str = "_"

    print "generate ngrams for question1"
    df.loc[:,"question1_unigram"] = list(map(preprocess_data, df["question1"]))   
    df.loc[:,"question1_bigram"] = [ngram.getBigram(x, join_str) for x in df["question1_unigram"]]
    df.loc[:,"question1_trigram"] = [ngram.getTrigram(x, join_str) for x in df["question1_unigram"]]

    print "generate ngrams for question2"
    
    df.loc[:,"question2_unigram"] = list(map(preprocess_data, df["question2"]))
    df.loc[:,"question2_bigram"] = [ngram.getBigram(x, join_str) for x in df["question2_unigram"]]
    df.loc[:,"question2_trigram"] = [ngram.getTrigram(x, join_str) for x in df["question2_unigram"]]

    
    ## jaccard coef/dice dist of n-gram
    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["question1", "question2"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names)-1):
                for j in range(i+1,len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                            map(partial(compute_dist, dist=dist), df[target_name+"_"+gram], df[obs_name+"_"+gram])
                           # list(df.apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))


    


###########################################
## Extract statistical distance features ##
###########################################
## generate dist stats feat
def generate_dist_stats_feat(dist, X_train, ids_train, X_test, ids_test, indices_dict, qids_test=None):

    stats_feat = 0 * np.ones((len(ids_test), stats_feat_num*config.n_classes), dtype=float)
    ## pairwise dist
    distance = pairwise_dist(X_test, X_train, dist)
    for i in range(len(ids_test)):
        id = ids_test[i]
        if qids_test is not None:
            qid = qids_test[i]
        for j in range(config.n_classes):
            key = (qid, j+1) if qids_test is not None else j+1
            if indices_dict.has_key(key):
                inds = indices_dict[key]
                # exclude this sample itself from the list of indices
                inds = [ ind for ind in inds if id != ids_train[ind] ]
                distance_tmp = distance[i][inds]
                if len(distance_tmp) != 0:
                    feat = [ func(distance_tmp) for func in stats_func ]
                    ## quantile
                    distance_tmp = pd.Series(distance_tmp)
                    quantiles = distance_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    stats_feat[i,j*stats_feat_num:(j+1)*stats_feat_num] = feat
    return stats_feat


def extract_statistical_distance_feat(path, dfTrain, dfTest, mode, feat_names):

    new_feat_names = copy(feat_names)
    ## get the indices of pooled samples
    relevance_indices_dict = get_sample_indices_by_relevance(dfTrain)
    query_relevance_indices_dict = get_sample_indices_by_relevance(dfTrain, "qid")
    ## very time consuming
    for dist in ["jaccard_coef", "dice_dist"]:
        for name in ["title", "description"]:
            for gram in ["unigram", "bigram", "trigram"]:
                ## train
                dist_stats_feat_by_relevance_train = generate_dist_stats_feat(dist, dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                            dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                            relevance_indices_dict)
                dist_stats_feat_by_query_relevance_train = generate_dist_stats_feat(dist, dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                                dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                                query_relevance_indices_dict, dfTrain["qid"].values)
                with open("%s/train.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, name, gram, dist), "wb") as f:
                    cPickle.dump(dist_stats_feat_by_relevance_train, f, -1)
                with open("%s/train.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, name, gram, dist), "wb") as f:
                    cPickle.dump(dist_stats_feat_by_query_relevance_train, f, -1)
                ## test
                dist_stats_feat_by_relevance_test = generate_dist_stats_feat(dist, dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                            dfTest[name+"_"+gram].values, dfTest["id"].values,
                                                            relevance_indices_dict)
                dist_stats_feat_by_query_relevance_test = generate_dist_stats_feat(dist, dfTrain[name+"_"+gram].values, dfTrain["id"].values,
                                                                dfTest[name+"_"+gram].values, dfTest["id"].values,
                                                                query_relevance_indices_dict, dfTest["qid"].values)
                with open("%s/%s.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, mode, name, gram, dist), "wb") as f:
                    cPickle.dump(dist_stats_feat_by_relevance_test, f, -1)
                with open("%s/%s.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, mode, name, gram, dist), "wb") as f:
                    cPickle.dump(dist_stats_feat_by_query_relevance_test, f, -1)

                ## update feat names
                new_feat_names.append( "%s_%s_%s_stats_feat_by_relevance" % (name, gram, dist) )
                new_feat_names.append( "%s_%s_%s_stats_feat_by_query_relevance" % (name, gram, dist) )

    return new_feat_names


##########
## Main ##
##########
if __name__ == "__main__":

    print sys.argv[1:2]
    start_time = time.time()

    if sys.argv[1:2][0] == "Training":
   
       ###############
       ## Load Data ##
       ###############
       ## load data
       with open(config.processed_train_data_path, "rb") as f:
           dfTrain = cPickle.load(f)
       ## load pre-defined stratified k-fold index
       with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
               skf = cPickle.load(f)
   
       ## file to save feat names
       feat_name_file = "%s/distance.feat_name" % config.feat_folder

   
       ## stats to extract
       quantiles_range = np.arange(0, 1.5, 0.5)
       stats_func = [ np.mean, np.std ]
       stats_feat_num = len(quantiles_range) + len(stats_func)
   
       #######################
       ## Generate Features ##
       #######################
       print("==================================================")
       print("Generate distance features...")
   
       extract_basic_distance_feat(dfTrain)
       feat_names = [name for name in dfTrain.columns if "jaccard_coef" in name or "dice_dist" in name]
                  
       for feat_name in feat_names:
           X_train = dfTrain[feat_name].values
           path = "%s/All" % config.feat_folder
           with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
               cPickle.dump(X_train, f, -1)



       print("For cross-validation...")
       for run in range(config.n_runs):
           for fold, (trainInd, validInd) in enumerate(skf[run]):
               print("Run: %d, Fold: %d" % (run+1, fold+1))
               path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                 
               for feat_name in feat_names:
                   X_train = dfTrain[feat_name].values[trainInd]
                   X_valid = dfTrain[feat_name].values[validInd]
                   with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_train, f, -1)
                   with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_valid, f, -1)
                   print X_train.shape
                   print X_valid.shape
               ## extract statistical distance features
               if stats_feat_flag:
                   dfTrain2 = dfTrain.iloc[trainInd].copy()
                   dfValid = dfTrain.iloc[validInd].copy()
                   extract_statistical_distance_feat(path, dfTrain2, dfValid, "valid", feat_names)
 
       ## save feat names
       print("Feature names are stored in %s" % feat_name_file)
       ## dump feat name
       dump_feat_name(feat_names, feat_name_file)
       print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
       print("Done.")


    #===========================================
    ###For testing...
    print sys.argv[1]
    if sys.argv[1] == "Testing":
       if sys.argv[2] == "All":
           Ntest = range(config.test_subset_number)
       else:
           exec("Ntest ="+ sys.argv[2])
       path = "%s/All" % config.feat_folder
       ## use full version for X_train

       for i in Ntest:
           print "For Testing set No. %s subset" % str(i)
           with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "rb") as f:
              subset_dfTest = cPickle.load(f)
           extract_basic_distance_feat(subset_dfTest)
           print subset_dfTest.shape
           feat_names = [name for name in subset_dfTest.columns if "jaccard_coef" in name or "dice_dist" in name]
    
           for feat_name in feat_names:
               X_test = subset_dfTest[feat_name].values
               with open("%s/test.%s.%s.feat.pkl" % (path, str(i),feat_name), "wb") as f:
                   cPickle.dump(X_test, f, -1)

           ## extract statistical distance features
           if stats_feat_flag:
              mode = "test.%s"%str(i)
              feat_names = extract_statistical_distance_feat(path, dfTrain, subset_dfTest, mode, feat_names)
   
           print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
       print("All Done.")
