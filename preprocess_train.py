#usr/bin/env python
# -*- coding: utf-8 -*-

"""
__file__

    preprocess.py

__description__

    This file preprocesses data.

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


dfTrain = pd.read_csv(config.original_train_data_path,encoding='utf8').fillna("")

# number of train samples
num_train = dfTrain.shape[0]

print num_train

print("Done.")

######################
## Pre-process Data ##
######################
print("Pre-process data...")


## clean text
clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)


dfTrain.loc[:,"question1"] = list(map(clean, dfTrain["question1"]))
dfTrain.loc[:,"question2"] = list(map(clean, dfTrain["question2"]))

print("Done.")


dfTrain.to_csv('./check_csv/check.train.csv', index=False, encoding='utf-8')

###############
## Save Data ##
###############
print("Save data...")

with open(config.processed_train_data_path, "wb") as f:
    cPickle.dump(dfTrain, f, -1)
    
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
