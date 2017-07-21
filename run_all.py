
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
cmd = "python ./preprocess.py"
os.system(cmd)

# #### generate kfold
cmd = "python ./gen_kfold.py"
os.system(cmd)

#######################
## Generate features ##
#######################
#### query id 
output = "stdbuf -oL "
#cmd = output +"python ./genFeat_id_feat.py > ./log_output/log_idfeat 2>./log_output/log_idfeat_err"
#os.system(cmd)


cmd = output +"python ./genFeat_question_feat.py > ./log_output/log_qestionfeat 2>./log_output/log_questionfeat_err"
os.system(cmd)

print "question feature done"
#### counting feat
cmd = output + "python ./genFeat_counting_feat.py > ./log_output/log_countingfeat 2>./log_output/log_countingfeat_err"
os.system(cmd)

print "counting feature done"
#### distance feat
cmd = output +  "python ./genFeat_distance_feat.py > ./log_output/log_distfeat 2>./log_output/log_distfeat_err" 
os.system(cmd)
print "distance feature done"

#### basic tfidf genFeat_basic_tfidf_feat_sim_hybrid.py
cmd = output + "python ./genFeat_basic_tfidf_feat_sim_hybrid.py > ./log_output/log_tfidffeat 2>./log_output/log_tfidffeat_err"
os.system(cmd)
print "basic tfidf feature done"
#stdbuf -oL python genFeat_basic_tfidf_feat_sim_hybrid.py "Training" > ./log_output/log_tfidffeat1 2>./log_output/log_tfidffeat_err

#### cooccurrence tfidf
cmd = output + "python ./genFeat_cooccurrence_tfidf_feat.py  > ./log_output/log_coocfeat 2>./log_output/log_coocfeat_err"
os.system(cmd)
print "coocurrence feature done"

print "feature extraction done"
'''
#####################
## Combine Feature ##
#####################
#### combine feat

cmd = output + "python ./combine_feat_LSA_and_stats_feat_Low.py > ./log_output/log_combine_feat_[LSA_and_stats_feat_May19]_[Low] 2>./log_output/log_err_combine_feat_[LSA_and_stats_feat_May19]_[Low]"
print "Combining feature LSA_and_stats_feat_May19 Low"
os.system(cmd)


#### combine feat
cmd = output +  "python ./combine_feat_[LSA_svd150_and_Jaccard_coef_May19]_[Low].py > ./log_output/log_combine_feat_[LSA_svd150_and_Jaccard_coef_May19]_[Low] 2>./log_output/log_err_combine_feat_[LSA_svd150_and_Jaccard_coef_May19]_[Low]"
print "Combining feature LSA_svd150_and_Jaccard_coef Low"
os.system(cmd)

#### combine feat
cmd = output + "python ./combine_feat_[svd100_and_bow_May19]_[Low].py > ./log_output/log_combine_feat_[svd100_and_bow_May19]_[Low] 2> ./log_output/log_combine_feat_[svd100_and_bow_May19]_[Low]"
print "Combining feature svd100_and_bow_May19]_[Low]"
os.system(cmd)

#### combine feat
cmd = output + "python ./combine_feat_[svd100_and_bow_May19]_[High].py > ./log_output/log_combine_feat_[svd100_and_bow_May19]_[High] 2> ./log_output/log_err_combine_feat_[svd100_and_bow_May19]_[High]"
print "Combining feature svd100_and_bow_May19]_[High"
os.system(cmd)

