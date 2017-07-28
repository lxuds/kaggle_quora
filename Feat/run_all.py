
"""
__file__

	run_all.py

__description___
	
	This file generates all the features in one shot.

__author__

	Lei Xu < leixuast@gmail.com >

"""

import os

#################
## Preprocesss ##
#################
#### preprocess data


cmd = "python ./preprocess_test.py"
os.system(cmd)

cmd = "python ./preprocess_train.py"
os.system(cmd)


# #### generate kfold
cmd = "python ./gen_kfold.py"
os.system(cmd)

cmd = "python ./gen_info.py"
os.system(cmd)

#######################
## Generate features ##
#######################


if not os.path.exists('./log_output'):
    os.makedirs('./log_output')

####  question feature
cmd = "stdbuf -oL python ./genFeat_question_feat.py Training > ./log_output/log_qestionfeat_training 2>./log_output/log_questionfeat_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./genFeat_question_feat.py Testing > ./log_output/log_qestionfeat_training 2>./log_output/log_questionfeat_testing_err"
os.system(cmd)
print "question feature done"


#### counting feat
cmd = "stdbuf -oL python ./genFeat_counting_feat.py Training  > ./log_output/log_countingfeat_training 2>./log_output/log_countingfeat_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./genFeat_counting_feat.py Testing All > ./log_output/log_countingfeat_testing 2>./log_output/log_countingfeat_testing_err"
os.system(cmd)
print "counting feature done"

#### distance feat
cmd = "stdbuf -oL python ./genFeat_distance_feat.py Training > ./log_output/log_distfeat_training  2>./log_output/log_distfeat_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./genFeat_distance_feat.py Testing All > ./log_output/log_distfeat_testing  2>./log_output/log_distfeat_testing_err" 
os.system(cmd)
print "distance feature done"

#### tfidf/bow and SVD feat
cmd = "stdbuf -oL python ./genFeat_basic_tfidf_feat_sim_hybrid.py Training > ./log_output/log_tfidffeat_training 2>./log_output/log_tfidffeat_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./genFeat_basic_tfidf_feat_sim_hybrid.py Testing All > ./log_output/log_tfidffeat_testing 2>./log_output/log_tfidffeat_testing_err"
os.system(cmd)
print "basic tfidf feature done"

'''
# work in progress
#### cooccurrence tfidf
cmd = "stdbuf -oL python ./genFeat_cooccurrence_tfidf_feat.py Training  > ./log_output/log_coocfeat_training 2>./log_output/log_coocfeat_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./genFeat_cooccurrence_tfidf_feat.py Testing All  > ./log_output/log_coocfeat_training 2>./log_output/log_coocfeat_testing_err"
os.system(cmd)
print "coocurrence feature done"
'''
print "feature extraction done"


#####################
## Combine Feature ##
#####################
#### combine feat

print "Combining feature LSA_and_stats_feat_May19 Low"
cmd = "stdbuf -oL python ./combine_feat_LSA_and_stats_feat_low.py Training > ./log_output/log_combine_feat_LSA_and_stats_feat_Low_traning 2>./log_output/log_combine_feat_LSA_and_stats_feat_Low_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./combine_feat_LSA_and_stats_feat_low.py Testing All > ./log_output/log_combine_feat_LSA_and_stats_feat_Low_testing 2>./log_output/log_combine_feat_LSA_and_stats_feat_Low_testing_err"
os.system(cmd)
print "Combining feature LSA_and_stats_feat_May19 Low Done"

#### combine feat

print "Combining feature LSA_svd150_and_Jaccard_coef Low"
cmd = "stdbuf -oL python ./combine_feat_LSA_svd150_and_Jaccard_coef_low.py Training > ./log_output/log_combine_feat_LSA_svd150_and_Jaccard_coef_Low_training 2>./log_output/log_combine_feat_LSA_svd150_and_Jaccard_coef_Low_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./combine_feat_LSA_svd150_and_Jaccard_coef_low.py > ./log_output/log_combine_feat_LSA_svd150_and_Jaccard_coef_Low 2>./log_output/log_combine_feat_LSA_svd150_and_Jaccard_coef_Low_err"
os.system(cmd)
print "Combining feature LSA_svd150_and_Jaccard_coef Low Done"


#### combine feat
print "Combining feature svd100_and_bow_Low"
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_low.py Training > ./log_output/log_combine_feat_svd100_and_bow_Low_training 2> ./log_output/log_combine_feat_svd100_and_bow_Low_training_err"
os.system(cmd)
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_low.py Testing All> ./log_output/log_combine_feat_svd100_and_bow_Low 2> ./log_output/log_combine_feat_svd100_and_bow_Low_testing_err"
os.system(cmd)
print "Combining feature svd100_and_bow_Low Done"



#### combine feat

print "Combining feature svd100_and_bow_High"
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_high.py Training > ./log_output/log_combine_feat_svd100_and_bow_High 2> ./log_output/log_combine_feat_svd100_and_bow_High_traning_err"
os.system(cmd)
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_high.py Testing All > ./log_output/log_combine_feat_svd100_and_bow_High 2> ./log_output/log_combine_feat_svd100_and_bow_High_testing_err"
os.system(cmd)
print "Combining feature svd100_and_bow_High Done"
