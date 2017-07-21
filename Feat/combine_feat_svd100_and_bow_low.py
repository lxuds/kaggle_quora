
"""
__file__
    
    combine_feat_[svd100_and_bow_May19]_[Low].py

__description__

    This file generates one combination of feature set (Low).

__author__

    Lei Xu < leixuast@gmail.com >

"""

import sys
sys.path.append("../")
from param_config import config
#from gen_info import gen_info
from combine_feat import combine_feat, SimpleTransform

            
if __name__ == "__main__":

    feat_names = [

        ##############
        ## Query id ##
        ##############
#        ("qid", SimpleTransform()),

        ################
        ## Question   ##
        ################
        ('q1_who', SimpleTransform()),
        ('q2_who', SimpleTransform()),
        ('who_both', SimpleTransform()),
        ('q1_where', SimpleTransform()),
        ('q2_where', SimpleTransform()),
        ('where_both', SimpleTransform()),
        ('q1_when', SimpleTransform()),
        ('q2_when', SimpleTransform()),
        ('when_both', SimpleTransform()),
        ('q1_why', SimpleTransform()),
        ('q2_why', SimpleTransform()),
        ('why_both', SimpleTransform()),
        ('question1_can', SimpleTransform()),
        ('question2_can', SimpleTransform()),
        ('question1_could', SimpleTransform()),
        ('question2_could', SimpleTransform()),
        ('question1_may', SimpleTransform()),
        ('question2_may', SimpleTransform()),
        ('question1_will', SimpleTransform()),
        ('question2_will', SimpleTransform()),
        ('question1_would', SimpleTransform()),
        ('question2_would', SimpleTransform()),
        ('question1_is', SimpleTransform()),
        ('question2_is', SimpleTransform()),
        ('question1_are', SimpleTransform()),
        ('question2_are', SimpleTransform()),
        ('question1_am', SimpleTransform()),
        ('question2_am', SimpleTransform()),
        ('question1_shall', SimpleTransform()),
        ('question2_shall', SimpleTransform()),
        ('question1_should', SimpleTransform()),
        ('question2_should', SimpleTransform()),
        ('question1_does', SimpleTransform()),
        ('question2_does', SimpleTransform()),
        ('question1_did', SimpleTransform()),
        ('question2_did', SimpleTransform()),
        ('question1_has', SimpleTransform()),
        ('question2_has', SimpleTransform()),
        ('question1_have', SimpleTransform()),
        ('question2_have', SimpleTransform()),
        ('pos_of_1st_qmark_question1', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_1st_qmark_question1', SimpleTransform()),
        ('count_qmark_question1', SimpleTransform(config.count_feat_transform)),
        ('pos_of_1st_qmark_question2', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_1st_qmark_question2', SimpleTransform()),
        ('count_qmark_question2', SimpleTransform(config.count_feat_transform)),

        ################
        ## Word count ##
        ################
        ('count_of_question1_unigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question1_unigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question1_unigram', SimpleTransform()),
        ('count_of_question1_bigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question1_bigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question1_bigram', SimpleTransform()),
        ('count_of_question1_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question1_trigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question1_trigram', SimpleTransform()),
        ('count_of_digit_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_digit_in_question1', SimpleTransform()),
        ('count_of_question2_unigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question2_unigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question2_unigram', SimpleTransform()),
        ('count_of_question2_bigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question2_bigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question2_bigram', SimpleTransform()),
        ('count_of_question2_trigram', SimpleTransform(config.count_feat_transform)),
        ('count_of_unique_question2_trigram', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_unique_question2_trigram', SimpleTransform()),
        ('count_of_digit_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_digit_in_question2', SimpleTransform()),
        ('count_of_question1_unigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_unigram_in_question2', SimpleTransform()),
        ('count_of_question2_unigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_unigram_in_question1', SimpleTransform()),
        ('question2_unigram_in_question1_div_question1_unigram', SimpleTransform()),
        ('question2_unigram_in_question1_div_question1_unigram_in_question2', SimpleTransform()),
        ('count_of_question1_bigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_bigram_in_question2', SimpleTransform()),
        ('count_of_question2_bigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_bigram_in_question1', SimpleTransform()),
        ('question2_bigram_in_question1_div_question1_bigram', SimpleTransform()),
        ('question2_bigram_in_question1_div_question1_bigram_in_question2', SimpleTransform()),
        ('count_of_question1_trigram_in_question2', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question1_trigram_in_question2', SimpleTransform()),
        ('count_of_question2_trigram_in_question1', SimpleTransform(config.count_feat_transform)),
        ('ratio_of_question2_trigram_in_question1', SimpleTransform()),
        ('question2_trigram_in_question1_div_question1_trigram', SimpleTransform()),
        ('question2_trigram_in_question1_div_question1_trigram_in_question2', SimpleTransform()),

        ##############
        ## Position ##
        ##############
        ('pos_of_question2_unigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_unigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_unigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_unigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question2_unigram_in_question1_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_question2_unigram_in_question1_min', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_mean', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_median', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_max', SimpleTransform()),
        ('normalized_pos_of_question2_unigram_in_question1_std', SimpleTransform()),
        ('pos_of_question1_unigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_unigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_unigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_unigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        ('pos_of_question1_unigram_in_question2_std', SimpleTransform(config.count_feat_transform)),
        ('normalized_pos_of_question1_unigram_in_question2_min', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_mean', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_median', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_max', SimpleTransform()),
        ('normalized_pos_of_question1_unigram_in_question2_std', SimpleTransform()),
        # ('pos_of_question2_bigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_bigram_in_question1_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_question2_bigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_median', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_bigram_in_question1_std', SimpleTransform()),
        # ('pos_of_question1_bigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_bigram_in_question2_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_question1_bigram_in_question2_min', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_mean', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_median', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_bigram_in_question2_std', SimpleTransform()),
        # ('pos_of_question2_trigram_in_question1_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question2_trigram_in_question1_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_question2_trigram_in_question1_min', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_mean', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_median', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_max', SimpleTransform()),
        # ('normalized_pos_of_question2_trigram_in_question1_std', SimpleTransform()),
        # ('pos_of_question1_trigram_in_question2_min', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_mean', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_median', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_max', SimpleTransform(config.count_feat_transform)),
        # ('pos_of_question1_trigram_in_question2_std', SimpleTransform(config.count_feat_transform)),
        # ('normalized_pos_of_question1_trigram_in_question2_min', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_mean', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_median', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_max', SimpleTransform()),
        # ('normalized_pos_of_question1_trigram_in_question2_std', SimpleTransform()),

#        ('description_missing', SimpleTransform()),

        ## jaccard coef
        ('jaccard_coef_of_unigram_between_question1_question2', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_question1_question2', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_question1_question2', SimpleTransform()),

        ## dice dist
        ('dice_dist_of_unigram_between_question1_question2', SimpleTransform()),
        ('dice_dist_of_bigram_between_question1_question2', SimpleTransform()),
        ('dice_dist_of_trigram_between_question1_question2', SimpleTransform()),

        #########
        ## BOW ##
        #########
        # ('question1_bow_common_vocabulary', SimpleTransform()),
        # ('question2_bow_common_vocabulary', SimpleTransform()),
        #('question2_bow_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
#        ('question2_bow_common_vocabulary_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),
        ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_cosine_sim', SimpleTransform()),
        # # ('question1_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # # ('question2_bow_common_vocabulary_common_svd100', SimpleTransform()),
        # # ('question2_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('question2_bow_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),
        # # ('question1_bow_common_vocabulary_question2_bow_common_vocabulary_bow_common_svd100_cosine_sim', SimpleTransform()),
        # ('question1_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question2_bow_common_vocabulary_individual_svd100', SimpleTransform()),
        # # ('question2_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # # ('question2_bow_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),
        
        ############
        ## TF-IDF ##
        ############
        # ('question1_tfidf_common_vocabulary', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary', SimpleTransform()),
        #('question2_tfidf_common_vocabulary_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
#        ('question2_tfidf_common_vocabulary_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_common_svd100_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),
        # ('question1_tfidf_common_vocabulary_question2_tfidf_common_vocabulary_tfidf_common_svd100_cosine_sim', SimpleTransform()),
        ('question1_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        ('question2_tfidf_common_vocabulary_individual_svd100', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_relevance', SimpleTransform()),
        # ('question2_tfidf_common_vocabulary_individual_svd100_cosine_sim_stats_feat_by_question1_relevance', SimpleTransform()),

    ]
    

    #gen_info(feat_path_name="svd100_and_bow_low_May19")
    #combine_feat(feat_names, feat_path_name="svd100_and_bow_low_May19")    


    if sys.argv[1] == "Training":
        mode = "Training"
        combine_feat(feat_names, "svd100_and_bow_low_May19", mode)
    if sys.argv[1] == "Testing":
        mode = "Testing"
        if sys.argv[2] == "All":
           Ntest = range(config.test_subset_number)
        else:
           exec("Ntest ="+ sys.argv[2])
        combine_feat(feat_names, "svd100_and_bow_low_May19", mode, Ntest)




