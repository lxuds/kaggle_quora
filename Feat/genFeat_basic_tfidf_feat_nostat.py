
"""
__file__
    genFeat_basic_tfidf_feat_nostat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. basic tfidf features for question1/question2
            - use common vocabulary among question1/question2 for further computation of cosine similarity

        2. cosine similarity between question1/question2 pairs
            - just plain cosine similarity

        3. SVD version of the above features

__author__

    Lei Xu < leixuast@gmail.com >

"""
import gc
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import cPickle
import numpy as np
import pandas as pd
from copy import copy
from scipy.sparse import vstack
from scipy import sparse
from nlp_utils import getTFV, getBOW
from feat_utils import get_sample_indices_by_relevance, dump_feat_name
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
sys.path.append("../")
from param_config import config
import time
from datetime import timedelta
from numpy import dot
#from numpy.linalg import norm

#####################
## Helper function ##
#####################


## compute cosine similarity
def norm_matrix_row(x):
    return np.sqrt(np.sum(x.multiply(x), axis=1))

def cosine_similarity(a, b):
    cos_sim = np.zeros((a.shape[0],1))
    denominator = np.multiply(norm_matrix_row(a), norm_matrix_row(b))
    non_zero_norm_ind = denominator.nonzero()
    cos_sim[non_zero_norm_ind] = np.sum(a.multiply(b), axis=1)[non_zero_norm_ind]/denominator[non_zero_norm_ind]   
    return cos_sim


## generate vectorizer and save it. save the 
def generate_vectorizer(path, dfTrain, feat_names, column_names):
    ## first fit a bow/tfidf on the all_text to get
    ## the common vocabulary to ensure question1/question2    
    ## has the same length bow/tfidf for computing the similarity
    if vocabulary_type == "common":
        if vec_type == "tfidf":
            vec = getTFV(ngram_range=ngram_range)
        elif vec_type == "bow":
            vec = getBOW(ngram_range=ngram_range)
        vec.fit(dfTrain["all_text"])
        vocabulary = vec.vocabulary_
    elif vocabulary_type == "individual":
        vocabulary = None
    save_vec_path = "%s/vectorizer_train" % (path)
    if not os.path.exists(save_vec_path):
        os.makedirs(save_vec_path)


    ##########################
    ## basic bow/tfidf feat ##
    ##########################
    for feat_name,column_name in zip(feat_names, column_names):
        print "generate %s feat for training sets %s and save %s vectorizer" % (vec_type, column_name, vec_type)
        if vec_type == "tfidf":
            vec = getTFV(ngram_range=ngram_range, vocabulary=vocabulary)
        elif vec_type == "bow":
            vec = getBOW(ngram_range=ngram_range, vocabulary=vocabulary)
        X_train = vec.fit_transform(dfTrain[column_name])
        vec.stop_words_ = None

        save_vec = "%s/train.%s.%s.%s.vectorizer.pkl" %(save_vec_path, vocabulary_type, vec_type, column_name)
        with open(save_vec, 'wb') as fout:
            cPickle.dump(vec, fout)
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_train, f, -1)

    ##############################################
    ## cosine sim feat for basic bow/tfidf feat ##
    ##############################################
    for i in range(len(feat_names)-1):
        for j in range(i+1,len(feat_names)):
            print "generate common %s cosine sim feat for %s and %s" % (vec_type, feat_names[i], feat_names[j])
            for mod in ["train"]:
                with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[i]), "rb") as f:
                    target_vec = cPickle.load(f)
                with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[j]), "rb") as f:
                    obs_vec = cPickle.load(f)
                sim = np.array([cosine_sim(target_vec[k], obs_vec[k]) for k in range(target_vec.shape[0])])[:,np.newaxis]
                ## dump feat
                with open("%s/%s.%s_%s_%s_cosine_sim.feat.pkl" % (path, mod, feat_names[i], feat_names[j], vec_type), "wb") as f:
                    cPickle.dump(sim, f, -1)
    return

def generate_svd_transformer(path, dfTrain, feat_names, column_names):
    ##################
    ## SVD features ##
    ##################
    ## we fit svd use stacked question1/question2 bow/tfidf for further cosine simalirity computation
    save_vec_path = "%s/vectorizer_train" % (path)
    print "Generate common SVD transformers"
    for i,feat_name in enumerate(feat_names):
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
            X_vec_train = cPickle.load(f)
        if i == 0:
            X_vec_all_train = X_vec_train
        else:
            X_vec_all_train = vstack([X_vec_all_train, X_vec_train])

    for n_components in svd_n_components:
        print "common svd %d components:" % n_components
        svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=2017, algorithm='randomized')
        svd.fit(X_vec_all_train)
        save_svd = "%s/train.%s.%s.common_svd%d_vectorizer.pkl" %(save_vec_path, vocabulary_type, vec_type, n_components)
    return


def generate_individual_svd_transformer(path, dfTrain, feat_names, column_names):
        #########################
        ## Individual SVD feat ##
        #########################
        ## generate individual svd feat
    for n_components in svd_n_components:
        for feat_name,column_name in zip(feat_names, column_names):
            print "generate individual %s-svd%d feat for %s" % (vec_type, n_components, column_name)
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            svd = TruncatedSVD(n_components=n_components, n_iter=15,random_state=2017, algorithm='randomized')
            X_svd_train = svd.fit_transform(X_vec_train)

    return 


# load basic vectorizer
def load_basic_vectorizer(path, feat_name, column_name):
    print "load %s %s %s vectorizer" % (vec_type, feat_name, column_name) 
    save_vec_path = "%s/vectorizer_train" % (path)
    save_vec = "%s/train.%s.%s.%s.vectorizer.pkl" %(save_vec_path, vocabulary_type, vec_type, column_name)
    with open(save_vec, "rb") as f:
        vec = cPickle.load(f)
    return vec


# generate common svd transformer
def load_common_svd_transformer(path, feat_names, n_components):
    ## we fit svd use stacked question1/question2 bow/tfidf for further cosine simalirity computation
    #save_vec_path = "%s/vectorizer_train" % (path)
    print "Generate common SVD transformers"
    for i,feat_name in enumerate(feat_names):
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
            X_vec_train = cPickle.load(f)
        if i == 0:
            X_vec_all_train = X_vec_train
        else:
            X_vec_all_train = vstack([X_vec_all_train, X_vec_train])
    X_vec_all_train = sparse.csr_matrix(X_vec_all_train)
    
    svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=2017, algorithm='randomized')
    svd.fit(X_vec_all_train)
    return svd


'''
def load_individual_svd_transformer(path, dfTrain, feat_names, n_components):
    #########################
    ## Individual SVD feat ##
    #########################
    ## generate individual svd feat
    print "generate individual %s-svd%d feat for %s" % (vec_type, n_components, column_name)
    with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
        X_vec_train = cPickle.load(f)
    svd = TruncatedSVD(n_components=n_components, n_iter=15,random_state=2017, algorithm='randomized')
    #X_svd_train = svd.fit_transform(X_vec_train)
    svd.fit(X_vec_train)
    return
'''


## generate basic bow/tfidf feat for testing subsets
def extract_basic_vec_feat_testing(path, dfTest, mode, feat_name, column_name, vec):
    print "generate %s feat for %s %s" % (vec_type, mode, column_name)
    X_test = vec.transform(dfTest[column_name])
    with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
        cPickle.dump(X_test, f, -1)
    return

## generate basic bow/tfidf cosine similarity for testing subsets
def extract_basic_vec_cosine_sim_feat_testing(path, mode, feat_names):
    with open("%s/%s.%s.feat.pkl" % (path, mode, feat_names[0]), "rb") as f:
        target_vec = cPickle.load(f)
    with open("%s/%s.%s.feat.pkl" % (path, mode, feat_names[1]), "rb") as f:
        obs_vec = cPickle.load(f)
    sim = cosine_similarity(target_vec, obs_vec)
    ## dump feat
    with open("%s/%s.%s_%s_%s_cosine_sim.feat.pkl" % (path, mode, feat_names[0], feat_names[1], vec_type), "wb") as f:
        cPickle.dump(sim, f, -1)
    return

## generate common svd for testing subsets
def extract_common_svd_feat_testing(path, mode, feat_name, n_components, svd):    
    print "generate common %s-svd%d feat for %s %s" % (vec_type, n_components, mode, feat_name)
    ## load bow/tfidf 
    with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
        X_vec_test = cPickle.load(f)
    X_vec_test = sparse.csr_matrix(X_vec_test)
    X_svd_test = svd.transform(X_vec_test)
    with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
        cPickle.dump(X_svd_test, f, -1)
    return

## generate common svd cosine similarity for testing subsets
def extract_common_svd_cosine_sim_feat_testing(path, mode, feat_names, n_components):    
    print "generate common %s-svd%d cosine sim feat for %s %s and %s" % (vec_type, n_components, mode, feat_names[0], feat_names[1])
    with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mode, feat_names[0], n_components), "rb") as f:
        target_vec = cPickle.load(f)
    with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mode, feat_names[1], n_components), "rb") as f:
        obs_vec = cPickle.load(f)
    target_vec = sparse.csr_matrix(target_vec)
    obs_vec = sparse.csr_matrix(obs_vec)
    sim = cosine_similarity(target_vec, obs_vec)
    ## dump feat
    with open("%s/%s.%s_%s_%s_common_svd%d_cosine_sim.feat.pkl" % (path, mode, feat_names[0], feat_names[1], vec_type, n_components), "wb") as f:
        cPickle.dump(sim, f, -1)
    return

## generate individual svd feat
def extract_individual_svd_feat_testing(path, dfTest, mode, feat_names, column_names):        
    save_vec_path = "%s/vectorizer_train" % (path)
    for feat_name,column_name in zip(feat_names, column_names):
        print "generate individual %s-svd%d feat for %s" % (vec_type, n_components, column_name)
        with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
            X_vec_test = cPickle.load(f)

        save_svd = "%s/train.%s.%s.individual_svd%d_vectorizer.pkl" %(save_vec_path, vocabulary_type, vec_type, n_components)
        with open(save_svd, 'rb') as fout:
            svd = cPickle.load(fout)
        X_svd_test = svd.transform(X_vec_test)
        with open("%s/%s.%s_individual_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
            cPickle.dump(X_svd_test, f, -1)
    return 





## extract all features for cross validation 
def extract_feat(path, dfTrain, dfTest, mode, feat_names, column_names):
    new_feat_names = copy(feat_names)
    ## first fit a bow/tfidf on the all_text to get
    ## the common vocabulary to ensure question1/question2    
    ## has the same length bow/tfidf for computing the similarity

    if vocabulary_type == "common":
        if vec_type == "tfidf":
            vec = getTFV(ngram_range=ngram_range)
        elif vec_type == "bow":
            vec = getBOW(ngram_range=ngram_range)
        vec.fit(dfTrain["all_text"])
        vocabulary = vec.vocabulary_
    elif vocabulary_type == "individual":
        vocabulary = None
    for feat_name,column_name in zip(feat_names, column_names):

        ##########################
        ## basic bow/tfidf feat ##
        ##########################
        print "generate %s feat for %s\n" % (vec_type, column_name)

        save_vec_path = "%s/vectorizer_train" % (path)
        save_vec = "%s/train.%s.%s.%s.vectorizer.pkl" %(save_vec_path, vocabulary_type, vec_type, column_name)
        print "vectorizer:", save_vec
        with open(save_vec, "rb") as f:
            vec = cPickle.load(f)
        X_test = vec.transform(dfTest[column_name])
        with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
            cPickle.dump(X_test, f, -1)


    #####################
    ## cosine sim feat ##
    #####################
    for i in range(len(feat_names)-1):
        for j in range(i+1,len(feat_names)):
            print "generate common %s cosine sim feat for %s and %s" % (vec_type, feat_names[i], feat_names[j])
            for mod in ["train", mode]:
                with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[i]), "rb") as f:
                    target_vec = cPickle.load(f)
                with open("%s/%s.%s.feat.pkl" % (path, mod, feat_names[j]), "rb") as f:
                    obs_vec = cPickle.load(f)
                #sim = np.asarray(map(cosine_sim, target_vec, obs_vec))[:,np.newaxis]
                sim = np.array([cosine_sim(target_vec[k], obs_vec[k]) for k in range(target_vec.shape[0])])[:,np.newaxis]
 #               sim = np.asarray(map(cosine_sim, target_vec, obs_vec)).reshape(-1,1)

                ## dump feat
                with open("%s/%s.%s_%s_%s_cosine_sim.feat.pkl" % (path, mod, feat_names[i], feat_names[j], vec_type), "wb") as f:
                    cPickle.dump(sim, f, -1)
            ## update feat names
            new_feat_names.append( "%s_%s_%s_cosine_sim" % (feat_names[i], feat_names[j], vec_type))

    
    ##################
    ## SVD features ##
    ##################
    ## we fit svd use stacked question1/question2 bow/tfidf for further cosine simalirity computation
    for i,feat_name in enumerate(feat_names):
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
            X_vec_train = cPickle.load(f)
        if i == 0:
            X_vec_all_train = X_vec_train
        else:
            X_vec_all_train = vstack([X_vec_all_train, X_vec_train])

    for n_components in svd_n_components:
        svd = TruncatedSVD(n_components=n_components, n_iter=15)
        svd.fit(X_vec_all_train)
        ## load bow/tfidf (for less coding...)
        for feat_name,column_name in zip(feat_names, column_names):
            print "generate common %s-svd%d feat for %s" % (vec_type, n_components, column_name)
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
                X_vec_test = cPickle.load(f)
            X_svd_train = svd.transform(X_vec_train)
            X_svd_test = svd.transform(X_vec_test)

            with open("%s/train.%s_common_svd%d.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_test, f, -1)

            ## update feat names
            new_feat_names.append( "%s_common_svd%d" % (feat_name, n_components) )
            
        
        #####################
        ## cosine sim feat ##
        #####################
        for i in range(len(feat_names)-1):
            for j in range(i+1,len(feat_names)):
                print "generate common %s-svd%d cosine sim feat for %s and %s" % (vec_type, n_components, feat_names[i], feat_names[j])
                for mod in ["train", mode]:
                    with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mod, feat_names[i], n_components), "rb") as f:
                        target_vec = cPickle.load(f)
                    with open("%s/%s.%s_common_svd%d.feat.pkl" % (path, mod, feat_names[j], n_components), "rb") as f:
                        obs_vec = cPickle.load(f)
                    sim = np.array([cosine_sim(target_vec[k], obs_vec[k]) for k in range(target_vec.shape[0])])[:,np.newaxis]
                    #sim = np.asarray(map(cosine_sim, target_vec, obs_vec))[:,np.newaxis]
                    ## dump feat
                    with open("%s/%s.%s_%s_%s_common_svd%d_cosine_sim.feat.pkl" % (path, mod, feat_names[i], feat_names[j], vec_type, n_components), "wb") as f:
                        cPickle.dump(sim, f, -1)
                ## update feat names
                new_feat_names.append( "%s_%s_%s_common_svd%d_cosine_sim" % (feat_names[i], feat_names[j], vec_type, n_components))
 
        
        #########################
        ## Individual SVD feat ##
        #########################
        ## generate individual svd feat
        for feat_name,column_name in zip(feat_names, column_names):
            print "generate individual %s-svd%d feat for %s" % (vec_type, n_components, column_name)
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "rb") as f:
                X_vec_train = cPickle.load(f)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "rb") as f:
                X_vec_test = cPickle.load(f)
            svd = TruncatedSVD(n_components=n_components, n_iter=15)
            X_svd_train = svd.fit_transform(X_vec_train)
            X_svd_test = svd.transform(X_vec_test)
            with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_individual_svd%d.feat.pkl" % (path, mode, feat_name, n_components), "wb") as f:
                cPickle.dump(X_svd_test, f, -1)
            ## update feat names
            new_feat_names.append( "%s_individual_svd%d" % (feat_name, n_components) )
    return new_feat_names



def cat_text(x1,x2):
    res = ' %s %s ' % (x1, x2)    
    return res


if __name__ == "__main__":

    print sys.argv[1]
    start_time = time.time()
    ## for fitting common vocabulary

    ############
    ## Config ##
    ############

    ## tfidf config
    vec_types = [ "tfidf", "bow" ]
    ngram_range = config.basic_tfidf_ngram_range
    vocabulary_type = config.basic_tfidf_vocabulary_type
    svd_n_components = [100, 150]
    tsne_n_components = [2]

    ## feat name config
    column_names = [ "question1", "question2"]

    ## =============================================
    if sys.argv[1] == "Training":
        ## load data
        print "Loading data\n"
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = cPickle.load(f)
        dfTrain["all_text"] = list(map(cat_text, dfTrain["question1"], dfTrain["question2"]))
        print "Train size:", dfTrain.shape[0]

        ## load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
               skf = cPickle.load(f)

        print "Done loading data\n"

        for vec_type in vec_types:
            print "Processing vector %s\n" % vec_type
            ## save feat names
            feat_names = [ "question1", "question2" ]
            feat_names = [ name+"_%s_%s_vocabulary" % (vec_type, vocabulary_type) for name in feat_names ]
            ## file to save feat names
            feat_name_file = "%s/basic_%s_and_cosine_sim.feat_name" % (config.feat_folder, vec_type)
 
            #######################
            ## Generate Features ##
            #######################
            print("==================================================\n")
            print("Generate basic %s features...\n" % vec_type)
 
            print("For cross-validation...")
            for run in range(config.n_runs):
                for fold, (trainInd, validInd) in enumerate(skf[run]):
            #for run in [0]:#range(config.n_runs):
            #    for fold, (trainInd, validInd) in enumerate(skf[run]):
                    print("Run: %d, Fold: %d" % (run+1, fold+1))
                    path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                    
                    dfTrain2 = dfTrain.iloc[trainInd].copy()
                    dfValid = dfTrain.iloc[validInd].copy()
                    print dfTrain2.shape, dfValid.shape
                    ## extract feat
                    new_feat_names = extract_feat(path, dfTrain2, dfValid, "valid", feat_names, column_names)
                    print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
            ## dump feat name
            dump_feat_name(new_feat_names, feat_name_file) 
            print("Done.")



    #===========================================
    ###For testing...
    if sys.argv[1] == "Testing":
       if sys.argv[2] == "All":
           Ntest = range(config.test_subset_number)
       else:
           exec("Ntest ="+ sys.argv[2])

       path = "%s/All" % config.feat_folder

       for vec_type in vec_types:
           print "*** Processing vector: %s" % vec_type
           feat_names = [ "question1", "question2" ]
           feat_names = [ name+"_%s_%s_vocabulary" % (vec_type, vocabulary_type) for name in feat_names ]
           
           for feat_name,column_name in zip(feat_names, column_names):
               vec = load_basic_vectorizer(path, feat_name, column_name)
               for i in Ntest:
                   #print "For Testing set No. %s subset" % str(i)
                   with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "rb") as f:
                       subset_dfTest = cPickle.load(f)
                   mode = "test.%s"%str(i)
                   extract_basic_vec_feat_testing(path, subset_dfTest, mode, feat_name, column_name, vec)

           print ("------------------------------------------")
           print vec_type, "**svd features"
           for n_components in svd_n_components:
               print "## n_components:", n_components
               svd_vec = load_common_svd_transformer(path, feat_names, n_components)
               #svd_individual_vec =  load_individual_svd_transformer()
               for feat_name,column_name in zip(feat_names, column_names):
                   for i in Ntest:
                       with open("%s.%s.pkl"%(config.processed_test_data_path,str(i)), "rb") as f:
                           subset_dfTest = cPickle.load(f)
                       mode = "test.%s"%str(i)
                       extract_common_svd_feat_testing(path, mode, feat_name, n_components, svd_vec)
                       gc.collect()
                       #extract_individual_svd_feat_testing(path, subset_dfTest, mode, feat_names, column_names)
           
           #cosine-similarity
           print ("------------------------------------------")
           print "generate cosine-similarity"
           for i in Ntest:
               mode = "test.%s"%str(i)
               extract_basic_vec_cosine_sim_feat_testing(path, mode, feat_names)
               for n_components in svd_n_components:
                   extract_common_svd_cosine_sim_feat_testing(path, mode, feat_names, n_components)

           print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
           print ("------------------------------------------")
       print("All Done.") 



    #=============================================
    # generate vectorizer using all training sets
    if sys.argv[1] == "vectorizer":
        print "Loading data"
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = cPickle.load(f)
        dfTrain["all_text"] = list(map(cat_text, dfTrain["question1"], dfTrain["question2"]))
        print "Train size:", dfTrain.shape[0]

        path = "%s/All" % config.feat_folder
        #dfTrain_test = dfTrain.loc[0:5000,:]
        for vec_type in vec_types:
            print "Processing vector %s" % vec_type
            print ("------------------------------------------")
            feat_names = [ "question1", "question2" ]
            feat_names = [ name+"_%s_%s_vocabulary" % (vec_type, vocabulary_type) for name in feat_names ]
           
            print("Generating Vectorizer for %s....and save features for whole training sets") %(vec_type)
            ## extract feat
            generate_vectorizer(path, dfTrain, feat_names, column_names)

            print ("----time elapsed----", str(timedelta(seconds=time.time() - start_time)))
            print ("------------------------------------------")
        print("All Done.")
        
