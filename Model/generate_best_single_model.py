
"""
__file__

	generate_best_single_model.py

__description__

	This file generates the best single model.

__author__

	Lei Xu < leixuast@gmail.com >

"""

import os

feat_names = [
	## svd100_and_bow_Jun27 (High)
	#"[Pre@solution]_[Feat@svd100_and_bow_Jun27]_[Model@reg_xgb_linear]",
	
	# ## you can also try the following models
	# "[Pre@solution]_[Feat@svd100_and_bow_Jun27]_[Model@cocr_xgb_linear]",
	# "[Pre@solution]_[Feat@svd100_and_bow_Jun27]_[Model@kappa_xgb_linear]",
	#"[Pre@solution]_[Feat@svd100_and_bow_high_May19]_[Model@reg_skl_ridge]",
        #"[Pre@solution]_[Feat@LSA_and_stats_feat_May19]_[Model@reg_skl_rf]",
        #"[Pre@solution]_[Feat@LSA_and_stats_feat_May19]_[Model@reg_xgb_tree]",
        #"[Pre@solution]_[Feat@LSA_and_stats_feat_May19]_[Model@reg_xgb_linear]",
] 

for feat_name in feat_names:
        
	#cmd = "stdbuf -oL python ./train_model.py %s > ./log/log%s 2>./log/errlog%s " % (feat_name, feat_name, feat_name) 
        #print cmd
        cmd = "python ./train_model.py " + feat_name
	os.system( cmd )
