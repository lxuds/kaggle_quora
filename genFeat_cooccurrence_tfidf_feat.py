
"""
__file__

    genFeat_cooccurrence_tfidf.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. tfidf for the following cooccurrence terms
            question1 unigram/bigram & question2 unigram/bigram
        2. corresponding lsa (svd) version features

__author__

    Lei Xu < leixuast@gmail.com >

"""

import re
import sys
import cPickle
import ngram
from feat_utils import dump_feat_name
from sklearn.decomposition import TruncatedSVD
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
sys.path.append("../")
from param_config import config
import time
from datetime import timedelta


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


########################
## Cooccurrence terms ##
########################
def cooccurrence_terms(lst1, lst2):
    join_str= "X"
    terms = [""] * len(lst1) * len(lst2)
    cnt =  0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res


##################
## Extract feat ##
##################
def extract_feat(df):
   
    print "generate ngrams"
    join_str = "_"

    print "generate ngrams for question1"
    df.loc[:,"question1_unigram"] = list(map(preprocess_data, df["question1"]))   
    df.loc[:,"question1_bigram"] = [ngram.getBigram(x, join_str) for x in df["question1_unigram"]]
#    df.loc[:,"question1_trigram"] = [ngram.getTrigram(x, join_str) for x in df["question1_unigram"]]

    print "generate ngrams for question2"
    
    df.loc[:,"question2_unigram"] = list(map(preprocess_data, df["question2"]))
    df.loc[:,"question2_bigram"] = [ngram.getBigram(x, join_str) for x in df["question2_unigram"]]
#    df.loc[:,"question2_trigram"] = [ngram.getTrigram(x, join_str) for x in df["question2_unigram"]]


    
    ## cooccurrence terms
#    join_str = "X"
    print "generate coocurance terms"
    df["question1_unigram_question2_unigram"] = map(cooccurrence_terms, df["question1_unigram"], df["question2_unigram"])
    df["question1_unigram_question2_bigram"] = map(cooccurrence_terms, df["question1_unigram"], df["question2_bigram"])

    # query bigram
    df["question1_bigram_question2_unigram"] = map(cooccurrence_terms, df["question1_bigram"], df["question2_unigram"])
    df["question1_bigram_question2_bigram"] = map(cooccurrence_terms, df["question1_bigram"], df["question2_bigram"])
      

if __name__ == "__main__":

    start_time = time.time()

    print sys.argv[1]


    ###########
    ## Config ##
    ############
    ## cooccurrence terms column names
    column_names = [
        "question1_unigram_question2_unigram",
        "question1_unigram_question2_bigram",
        "question1_bigram_question2_unigram",
        "question1_bigram_question2_bigram",
    ]
    ## feature names
    feat_names = [ name+"_tfidf" for name in column_names ]
    ## file to save feat names
    feat_name_file = "%s/intersect_tfidf.feat_name" % config.feat_folder

    ngram_range = config.cooccurrence_tfidf_ngram_range

    svd_n_components = 100

     ## load data
    with open(config.processed_train_data_path, "rb") as f:
         dfTrain = cPickle.load(f)
     ## get cooccurrence terms
    print "Generating cooccurrence terms for traning sets"
    extract_feat(dfTrain)
 

    if sys.argv[1] == "Training":
   
       ###############
       ## Load Data ##
       ###############
   
   
       ## load pre-defined stratified k-fold index
       with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
               skf = cPickle.load(f)
   
       #######################
       ## Generate Features ##
       #######################
       print("==================================================")
       print("Generate co-occurrence tfidf features...")
   
  
       ######################
       # Cross validation ##
       ######################
       print("For cross-validation...")
       for run in range(config.n_runs):
           for fold, (trainInd, validInd) in enumerate(skf[run]):
               print("Run: %d, Fold: %d" % (run+1, fold+1))
               path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                   
               for feat_name,column_name in zip(feat_names, column_names):
                   print "generate %s feat" % feat_name
                   ## tfidf
                   tfv = getTFV(ngram_range=ngram_range)
                   X_tfidf_train = tfv.fit_transform(dfTrain.iloc[trainInd][column_name])
                   X_tfidf_valid = tfv.transform(dfTrain.iloc[validInd][column_name])
                   with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_tfidf_train, f, -1)
                   with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                       cPickle.dump(X_tfidf_valid, f, -1)
   
                   ## svd
                   svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
                   X_svd_train = svd.fit_transform(X_tfidf_train)
                   X_svd_test = svd.transform(X_tfidf_valid)
                   with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                       cPickle.dump(X_svd_train, f, -1)
                   with open("%s/valid.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                       cPickle.dump(X_svd_test, f, -1)
   
       print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
   
       print("Done.")
       ## save feat names
       print("Feature names are stored in %s" % feat_name_file)
       feat_names += [ "%s_individual_svd%d"%(f, svd_n_components) for f in feat_names ]
       dump_feat_name(feat_names, feat_name_file)




    #===========================================
    ###For testing...
    if sys.argv[1] == "Testing":
       exec("Ntest ="+ sys.argv[2])
   
       #################
       ## Re-training ##
       #################
       print("==================================================")
       #print("Generate co-occurrence tfidf features for testing sets")   

       path = "%s/All" % config.feat_folder
           ## extract feat
       for i in Ntest:
           print "For Testing set No. %s subset" % str(i)
           with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "rb") as f:
               subset_dfTest = cPickle.load(f)
           print "Generate cooccurrence terms for testing sets"
           extract_feat(subset_dfTest)
                                                          
           for feat_name,column_name in zip(feat_names, column_names):
               print "generate %s feat" % feat_name
               tfv = getTFV(ngram_range=ngram_range)
               X_tfidf_train = tfv.fit_transform(dfTrain[column_name])
               X_tfidf_test = tfv.transform(subset_dfTest[column_name])
               with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                   cPickle.dump(X_tfidf_train, f, -1)
               with open("%s/test.%s.%s.feat.pkl" % (path,i,feat_name), "wb") as f:
                   cPickle.dump(X_tfidf_test, f, -1)
               print "%s/test.%s.%s.feat.pkl" % (path,i,feat_name)
               ## svd
               svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
               X_svd_train = svd.fit_transform(X_tfidf_train)
               X_svd_test = svd.transform(X_tfidf_test)
               with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                   cPickle.dump(X_svd_train, f, -1)
               with open("%s/test.%s.%s_individual_svd%d.feat.pkl" % (path,i, feat_name, svd_n_components), "wb") as f:
                   cPickle.dump(X_svd_test, f, -1)
       
       print("Done.")
    print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))

    print("All Done.")
