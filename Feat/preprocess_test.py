#usr/bin/env python
# -*- coding: utf-8 -*-

"""
__file__

    preprocess_test.py

__description__

    This file preprocesses testing sample data.

__author__

    Lei Xu < leixuast@gmail.com >
    
"""
import sys
import cPickle
import numpy as np
import pandas as pd
from nlp_utils import clean_text, pos_tag_text, convert_currency_numbers
sys.path.append("../")
from param_config import config
reload(sys)
sys.setdefaultencoding('utf-8')
from textacy.preprocess import replace_currency_symbols, normalize_whitespace
import re
#from feat_utils import try_divide, dump_feat_name

###############
## Load Data ##
###############
print("Load data...")


#dfTrain = pd.read_csv(config.original_train_data_path,encoding='utf8').fillna("")
dfTest = pd.read_csv(config.original_test_data_path,encoding='utf8').fillna("")

# number of test samples
num_test =  dfTest.shape[0]
print  num_test

print("Done.")


######################
## Pre-process Data ##
######################
print("Pre-process data...")

## insert fake label for test
dfTest["is_duplicate"] = np.zeros((num_test), dtype=np.int)
dfTest.rename(columns={"test_id": "id"}, inplace=True)



## clean text
clean = lambda line: clean_text(line,drop_html_flag=config.drop_html_flag)

#split testing semple into subset with size 100000

split_size = config.test_split_size
n_part = dfTest.shape[0] / split_size
if n_part*split_size < dfTest.shape[0]: n_part = n_part + 1
print "test splits: ", n_part
for i in range(n_part):
    if i == n_part-1:
       subset_dfTtest = dfTest.iloc[i*split_size:,:]
    else:
       subset_dfTest = dfTest.iloc[i*split_size:(i+1)*split_size,:]
    print i
    subset_dfTest.loc[:,"question1"] = list(map(clean, subset_dfTest["question1"]))
    subset_dfTest.loc[:,"question2"] = list(map(clean, subset_dfTest["question2"]))    
    with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "wb") as f:
         print "%s.%s.pkl"%(config.processed_test_data_path,str(i))
         cPickle.dump(subset_dfTest, f, -1)
    print subset_dfTest.shape[0]

    subset_dfTest.to_csv('./check_csv/check.test.%s.csv'%(str(i)) , index=False, encoding='utf-8')
print("Done.")


###############
## Save Data ##
###############
print("Save data...")

print("Done.")

"""
## pos tag text
dfTrain = dfTrain.apply(pos_tag_text, axis=1)
dfTest = dfTest.apply(pos_tag_text, axis=1)
with open(config.pos_tagged_train_data_path, "wb") as f:
    cPickle.dump(dfTrain, f, -1)
with open(config.pos_tagged_test_data_path, "wb") as f:
    cPickle.dump(dfTest, f, -1)
print("Done.")
"""
