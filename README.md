In this Quora Question Pairs competition, we were asked to build an advanced model to classify whether question pairs are duplicates or not.

Data: The traning sets include 404,290 labeled question pairs, and the testing sets include 2,345,796 question pairs. You may download the data at https://www.kaggle.com/c/quora-question-pairs.

The solution includes two sections: feature engineering (in Feat) and model ensembling (in Model). The feature engineering part was invloved with various natural language processing techniques. In model ensembling section, we employed Hyperopt package to find the best parameter settings for various algorithms, and built a model library with thousands of models of various hyper-parameters. We finally used bagging ensemble selection to find the best model ensemble from the model library.

Before feature engineering, we first preprocessed the data. We performed word replacement and alignments, e.g., replacing contractions, currency symbols, and units with standard forms. We removed accent and punctuation. We also performed Porter stemming before generating word counting and BOW/TF-IDF features. The NLTK (Natural Language Toolkit and Scikit-learn) and Regular Expression module in Python were heavily used in this process.

We then proceeded to extract and select features:
- Counting features, including the count and ratio of unigram/bigram/trigram, digit, unique n-gram, and intersect n-gram
- Distance Features: plain jaccard coefficient/dice distance for the unigram/bigram/trigram of each question pair
- Basic TF-IDF/BOW features for each question pair: We used common vocabulary among question1/question2 for further computation of cosine similarity. We computed the plain cosine similarity for each question pair, and SVD version of the above features
- Question features: count of interrogative words, auxiliary verbs and question marks
- cooccurrence features: TF-IDF for the following cooccurrence terms: question1 unigram/bigram & question2 unigram/bigram (work in progress)

In this section, NLTK and Sklearn packages were employed.

In model ensembling selection, we first employed Hyperopt package (a parameter searching algorithm) to build a model library with models of various algorithms (e.g., SVM, GBDT, RF), or the same models of various hyper-parameters. The hyper-parameter searching process not only found the best hyper-parameter for the model (corresponding to the best single model), but also helped build a model library with model of various parameters that can later be used in ensemble selection to further boost the performance of the best single model. We then used bagging ensemble selection to find the best model ensemble, and make prediction on the testing sets.

Code Description: 

In the folder Feat:

- preprocess_test.py: This file preprocesses testing sets.
- preprocess_train.py: This file preprocesses training sets.
- genFeat_counting_feat.py: This file generates counting features.
- genFeat_distance_feat.py: This file generates distance features.
- genFeat_basic_tfidf_feat_nostat.py: This file generates basic TF-IDF/BOW features, and LSA/SVD features.
- genFeat_question_feat.py: This file generates question features.
- genFeat_cooccurrence_tfidf_feat.py: This file generates cooccurrence features.
- gen_info.py: This file generates info file for each run/fold, and the whole training and testing sets.
- gen_kfold.py: This file generates the StratifiedKFold sample indices.
- combine_feat_LSA_and_stats_feat_low.py: This file generates one combination of feature set.
- combine_feat_LSA_svd150_and_Jaccard_coef_low.py: This file generates one combination of feature set.
- combine_feat_svd100_and_bow_high.py: This file generates one combination of feature set.
- combine_feat_svd100_and_bow_low.py: This file generates one combination of feature set.
- combine_feat.py: This file combines features and save them in svmlight format.
- feat_utils.py: This file providesutils for generating features.
- ngram.py: This file generates n-grams.
- nlp_utils.py: This file provides functions for NLP tasks.
- replacer.py: This file provides functions for words replacement.
- run_all.py: This file provides commands for preprocessing and generating features in one shot.

In the folder Model:

- generate_best_single_model.py: This file generates the best single model.
- train_model.py: This file trains single model, and generates models trained with various hyper-parameters.
- single_model_predict_test.py: This file used the trainig result (the best single model and models trained with various hyper-parameters) - to make predictions on testing sets
- generate_model_library.py: This file trains various models and generate the model library.
- ensemble_selection.py: This file trains the bagging ensemble model.
- generate_ensemble_train.py: This file trains the bagging ensemble model.
- ensemble_selection_predict.py: This file uses trained ensemble model to generate predictions on the testing sets.
- model_library_config.py: This file provides model library configurations for ensemble selection.
- utils.py: This file provides function to convert probabilities to class.
