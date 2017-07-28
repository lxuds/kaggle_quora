
"""
__file__

    genFeat_question_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. Interrogative Words Features
            
            1. Count of interrogative words

        2. Auxiliary Verbs Features

            1. Count of auxiliary verbs

        3. Question Marks

            1. Count of question marks

            2. Position/normalized position of the first question mark

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
from datetime import timedelta

from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, dump_feat_name
sys.path.append("../")
from param_config import config
from textacy.preprocess import replace_currency_symbols, normalize_whitespace


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


def add_word_count(x, df, word):
        x['q1_' + word] = df['question1'].apply(lambda q: (word in str(q).lower())*1)
        x['q2_' + word] = df['question2'].apply(lambda q: (word in str(q).lower())*1)
        x[word + '_both'] = x['q1_' + word] * x['q2_' + word]


def add_more_word_count(x, df):
    strings =[r"can", r"could", r"may", r"will", r"would", r"is", r"are", r"am", r"shall", r"should", r"does", r"did", r"has", r"have"]
    names = ["question1", "question2"]
    for string in strings:
        token_pattern = r"^" + string + r"(?:.*)|[\?.!;,:\'\"]\s*" + string+ r" (?:.*)"
        r = re.compile(token_pattern)
        for name in names:
            df.loc[:,name] = df[name].str.lower()
            x[name+"_"+string] = df[name].str.contains(r)
            x[name+"_"+string] = x[name+"_"+string].astype(int)
    return

########



def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


def question_mark_pos(line):
    """
       Get the list of positions of question mark in line
    """
    text = normalize_whitespace(line)
    token_pattern = r"\?{1,}\s{0,}\?{1,}"
    text = re.sub(token_pattern, r"?",line)  # replace multi question marks with just one
    #print line
    token_pattern = r"(?u)\b\w+\b|\?"
    r = re.compile(token_pattern)
    word_qmark = r.findall(line)
    #print word_qmark
    pos_qmark = get_position_list("?",word_qmark )
    count_qmark = len(pos_qmark)
    if pos_qmark[0] ==0:
       count_word = len(word_qmark)
       pos_qmark = [len(word_qmark)]
    else:
       count_word = len(word_qmark) - count_qmark
       pos_qmark = [i-j for j, i in enumerate(pos_qmark,start=1)]
    pos_qmark = pos_qmark + [count_word]
    return pos_qmark


def try_divide(x, y, val=0.0):
    """
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val



def extract_question_mark_feat(x, df):
    names = ["question1", "question2"]
    for name in names:
        x["pos_info_word_count_"+name] = list(map(question_mark_pos, df[name]))
        x["pos_qmark_"+name] = [y[:-1] for y in x["pos_info_word_count_"+name]] #map(question_mark_pos, df[name])
        x["count_word_"+name] = [y[-1] for y in x["pos_info_word_count_"+name]]
        x["pos_of_1st_qmark_"+name] = [y[0] for y in x["pos_info_word_count_"+name]]
#        print x["count_word_"+name]
#        print x["pos_of_1st_qmark_"+name]
        x["normalized_pos_of_1st_qmark_"+name] = map(try_divide, x["pos_of_1st_qmark_"+name], x["count_word_"+name])
        x["count_qmark_"+name] = [len(y)-1 for y in x["pos_info_word_count_"+name]]

        x.drop("pos_info_word_count_"+name, axis=1, inplace=True)
        x.drop("pos_qmark_"+name, axis=1, inplace=True)
        x.drop("count_word_"+name, axis=1, inplace=True)
    return


#def main():

if __name__ == "__main__":
    
    print sys.argv[1]   
    start_time = time.time()
    if sys.argv[1] == "Training":


       ###############
       ## Load Data ##
       ###############
       dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
       ## load pre-defined stratified k-fold index
       with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
                  skf = cPickle.load(f)
       ## file to save feat names
       feat_name_file = "%s/question.feat_name" % config.feat_folder
       
       #######################
       ## Generate Features ##
       #######################
       print("==================================================")
       print("Generate question features...")
       
        
       df = dfTrain #.iloc[0:19,:]
       x = pd.DataFrame()
       add_word_count(x, df,'how')
       add_word_count(x, df,'what')
       add_word_count(x, df,'which')
       add_word_count(x, df,'who')
       add_word_count(x, df,'where')
       add_word_count(x, df,'when')
       add_word_count(x, df,'why')
       
       add_more_word_count(x,df)
       
       extract_question_mark_feat(x,df)
       
       
       feat_names = list(x.columns.values)
       
       print feat_names   
       path = "%s/All" % config.feat_folder
       for feat_name in feat_names:
           X_train = x[feat_name].values
           #print X_train.shape
           with open("%s/train.%s.feat.pkl" % (path,feat_name), "wb") as f:
               print "%s/train.%s.feat.pkl" % (path, feat_name)
               cPickle.dump(X_train, f, -1)

       print("For cross-validation...")
       for run in range(config.n_runs):
           ## use 33% for training and 67 % for validation
           ## so we switch trainInd and validInd
           for fold, (trainInd, validInd) in enumerate(skf[run]):
               print("Run: %d, Fold: %d" % (run+1, fold+1))
               path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
               #########################
               ## get word count feat ##
               #########################
               for feat_name in feat_names:
                   X_train = x[feat_name].values[trainInd]
                   X_valid = x[feat_name].values[validInd]
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

    if sys.argv[1] == "Testing":   
       dfTest = pd.read_csv(config.original_test_data_path).fillna("")
       print("For testing...")
       path = "%s/All" % config.feat_folder
       ## use full version for X_train
       
       split_size = config.test_split_size
       n_part = dfTest.shape[0] / split_size
       if n_part*split_size < dfTest.shape[0]: n_part = n_part + 1
       print "test splits: ", n_part
       
       for i in range(n_part):
           if i == n_part-1:
              sub_set_dfTtest = dfTest.iloc[i*split_size:,:]
           else:
              sub_set_dfTest = dfTest.iloc[i*split_size:(i+1)*split_size,:]
    
           xx = pd.DataFrame()   
           add_word_count(xx, sub_set_dfTest,'how')
           add_word_count(xx, sub_set_dfTest,'what')
           add_word_count(xx, sub_set_dfTest,'which')
           add_word_count(xx, sub_set_dfTest,'who')
           add_word_count(xx, sub_set_dfTest,'where')
           add_word_count(xx, sub_set_dfTest,'when')
           add_word_count(xx, sub_set_dfTest,'why')
           add_more_word_count(xx,sub_set_dfTest)
           extract_question_mark_feat(xx,sub_set_dfTest)

           feat_names = list(xx.columns.values)
           print i
           print feat_names   
           for feat_name in feat_names:
               X_test = xx[feat_name].values
               #print X_test.shape[0]
               with open("%s/test.%s.%s.feat.pkl" % (path, str(i),feat_name), "wb") as f:
                   print "%s/test.%s.%s.feat.pkl" % (path, feat_name,str(i))
                   cPickle.dump(X_test, f, -1)
           print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
          
       print("All Done.")
       

