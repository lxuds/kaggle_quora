
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


output = "stdbuf -oL "
'''
cmd = "python ./preprocess_test.py"
os.system(cmd)

# #### generate kfold
cmd = "python ./gen_kfold.py"
os.system(cmd)

python gen_info.py

#######################
## Generate features ##
#######################
#### query id 
output = "stdbuf -oL "
#cmd = "stdbuf -oL python ./genFeat_id_feat.py > ./log_output/log_idfeat 2>./log_output/log_idfeat_err"
#os.system(cmd)


cmd = "stdbuf -oL python ./genFeat_question_feat.py Training > ./log_output/log_qestionfeat 2>./log_output/log_questionfeat_err"
os.system(cmd)


cmd = "stdbuf -oL python ./genFeat_question_feat.py Testing > ./log_output/log_qestionfeat 2>./log_output/log_questionfeat_err"
os.system(cmd)

print "question feature done"
#### counting feat
cmd = "stdbuf -oL python ./genFeat_counting_feat.py Testing All > ./log_output/log_countingfeat_testing 2>./log_output/log_countingfeat_testing_err"
os.system(cmd)

print "counting feature done"
#### distance feat
cmd = "stdbuf -oL python ./genFeat_distance_feat.py Testing All > ./log_output/log_distfeat_testing  2>./log_output/log_distfeat_testing_err" 
os.system(cmd)
print "distance feature done"

#### basic tfidf genFeat_basic_tfidf_feat_sim_hybrid.py
cmd = "stdbuf -oL python ./genFeat_basic_tfidf_feat_sim_hybrid.py Testing All > ./log_output/log_tfidffeat_testing 2>./log_output/log_tfidffeat_testing_err"
os.system(cmd)
print "basic tfidf feature done"
#stdbuf -oL python genFeat_basic_tfidf_feat_sim_hybrid.py "Training" > ./log_output/log_tfidffeat1 2>./log_output/log_tfidffeat_err

#### cooccurrence tfidf
cmd = "stdbuf -oL python ./genFeat_cooccurrence_tfidf_feat.py  > ./log_output/log_coocfeat 2>./log_output/log_coocfeat_err"
os.system(cmd)
print "coocurrence feature done"

print "feature extraction done"
'''
#####################
## Combine Feature ##
#####################
#### combine feat

cmd = "stdbuf -oL python ./combine_feat_LSA_and_stats_feat_low.py Testing All > ./log_output/log_combine_feat_LSA_and_stats_feat_Low 2>./log_output/log_err_combine_feat_LSA_and_stats_feat_Low"
print "Combining feature LSA_and_stats_feat_May19 Low"
os.system(cmd)


#### combine feat
cmd = "stdbuf -oL python ./combine_feat_LSA_svd150_and_Jaccard_coef_low.py > ./log_output/log_combine_feat_LSA_svd150_and_Jaccard_coef_Low 2>./log_output/log_err_combine_feat_LSA_svd150_and_Jaccard_coef_Low"
print "Combining feature LSA_svd150_and_Jaccard_coef Low"
os.system(cmd)

#### combine feat
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_low.py > ./log_output/log_combine_feat_svd100_and_bow_Low 2> ./log_output/log_combine_feat_svd100_and_bow_Low"
print "Combining feature svd100_and_bow_Low"
os.system(cmd)

#### combine feat
cmd = "stdbuf -oL python ./combine_feat_svd100_and_bow_high.py > ./log_output/log_combine_feat_svd100_and_bow_High 2> ./log_output/log_err_combine_feat_svd100_and_bow_High"
print "Combining feature svd100_and_bow_High"
os.system(cmd)

