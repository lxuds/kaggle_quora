
"""
__file__

    genFeat_counting_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. Basic Counting Features
            
            1. Count of n-gram in question1/question2

            2. Count & Ratio of Digit in question1/question2

            3. Count & Ratio of Unique n-gram in question1/question2

        2. Intersect Counting Features

            1. Count & Ratio of a's n-gram in b's n-gram

        3. Intersect Position Features

            1. Statistics of Positions of a's n-gram in b's n-gram

            2. Statistics of Normalized Positions of a's n-gram in b's n-gram

__author__

    Lei Xu < leixuast@gmail.com >

"""

import re
import sys
import ngram
import cPickle
import numpy as np
import pandas as pd
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, dump_feat_name
sys.path.append("../")
from param_config import config
import time
from datetime import timedelta



def get_position_list(obs, target):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def preprocess_data(line,
                    token_pattern=token_pattern,
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


def extract_feat(df):
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


    ################################
    ## word count and digit count ##
    ################################
    print "generate word counting features"
    feat_names = ["question1", "question2"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            ## word count
            df["count_of_%s_%s"%(feat_name,gram)] = [len(x) for x in df[feat_name+"_"+gram]]
            df["count_of_unique_%s_%s"%(feat_name,gram)] = [len(set(x)) for x in df[feat_name+"_"+gram]]
            df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        ## digit count
        df["count_of_digit_in_%s"%feat_name] = list(map(count_digit, df[feat_name+"_unigram"]))                                                               
        df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])


    ##############################
    ## intersect word count ##
    ##############################
    print "generate intersect word counting features"
    
    def word_count_intersect_questions(obs,target):
        word_count_intersect = 0
        if len(obs) != 0:
            word_count_intersect = len([w for w in obs if w in target])
        return word_count_intersect

    

    #### unigram
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    ## query
                    df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(map(word_count_intersect_questions, df[obs_name+"_"+gram], df[target_name+"_"+gram]))
                    df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

        ## some other feat
        df["question2_%s_in_question1_div_question1_%s"%(gram,gram)] = map(try_divide, df["count_of_question2_%s_in_question1"%gram], df["count_of_question1_%s"%gram])
        df["question2_%s_in_question1_div_question1_%s_in_question2"%(gram,gram)] = map(try_divide, df["count_of_question2_%s_in_question1"%gram], df["count_of_question1_%s_in_question2"%gram])


    ######################################
    ## intersect word position feat ##
    ######################################
    print "generate intersect word position features"
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = list(map(get_position_list, df[obs_name+"_"+gram], df[target_name+"_"+gram]))
                    ## stats feat on pos
                    df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                    df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                    df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                    df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                    df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                    ## stats feat on normalized_pos
                    df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])

                    
                    


if __name__ == "__main__":

    start_time = time.time()

    feat_name_file = "%s/counting.feat_name" % config.feat_folder
    print config.feat_folder
    print feat_name_file

    if sys.argv[1] == "Training":
   
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
       feat_name_file = "%s/counting.feat_name" % config.feat_folder

       
       #######################
       ## Generate Features ##
       #######################
       print("==================================================")
       print("Generate counting features...")

   
       extract_feat(dfTrain)
       feat_names = [
           name for name in dfTrain.columns \
               if "count" in name \
               or "ratio" in name \
               or "div" in name \
               or "pos_of" in name
       ]
       '''
       path = "%s/All" % config.feat_folder
       for feat_name in feat_names:
           X_train = dfTrain[feat_name].values
           with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
               cPickle.dump(X_train, f, -1)

       #print 'all_feature_names:', feat_names
       '''
       print("For cross-validation...")
       for run in range(config.n_runs):
           for fold, (trainInd, validInd) in enumerate(skf[run]):
               print("Run: %d, Fold: %d" % (run+1, fold+1))
               path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                 
               #########################
               ## get word count feat ##
               #########################
               for feat_name in feat_names:
                   X_train = dfTrain[feat_name].values[trainInd]
                   X_valid = dfTrain[feat_name].values[validInd]
                   print X_train.shape
                   print X_valid.shape
                   with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_train, f, -1)
                   with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_valid, f, -1)
       ## save feat names
       print("Feature names are stored in %s" % feat_name_file)
       ## dump feat name
       dump_feat_name(feat_names, feat_name_file)
       print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
       print("Done.")


    ## For testing...
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
           extract_feat(subset_dfTest)

           feat_names = [
               name for name in subset_dfTest.columns \
                 if "count" in name \
                 or "ratio" in name \
                 or "div" in name \
                 or "pos_of" in name
           ]


           for feat_name in feat_names:
               X_test = subset_dfTest[feat_name].values
               with open("%s/test.%s.%s.feat.pkl" % (path, str(i),feat_name), "wb") as f:
                  cPickle.dump(X_test, f, -1)
           print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))

       print("All Done.")
   
   
   
   
   
