import scipy.sparse as sp
import pandas as pd
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_double
from Recommenders.Hybrid.P3_RP3 import P3_RP3
from  Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.Hybrid.FirstLayer import FirstLayer
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Utils.Evaluator import EvaluatorHoldout
from Recommenders.Hybrid.RP3_ITEMHYBRID import RP3_SLIM_BPR



import os
from Utils import Reader
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname,  "data/interactions_and_impressions.csv")
matrix_extended=os.path.join(dirname,  "data/extended.csv")
ICM_path=os.path.join(dirname,  "data/data_ICM_type.csv")
ICM_path_length=os.path.join(dirname,  "data/data_ICM_length.csv")


URM_train=Reader.read_train_csr(matrix_path=matrix_path)
#URM_train_extended=Reader.read_train_csr_extended(matrix_path=matrix_extended)
#URM_train1, URM_test1, URM_train2, URM_test2 =split_train_in_two_percentage_global_sample_double(URM_train_extended,URM_train)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

evaluator = EvaluatorHoldout(URM_test_list=URM_validation, cutoff_list=[10], isRanking=False)


RECOMMENDER=P3_RP3(URM_train=URM_train)
RECOMMENDER.fit()
result_df, _ = evaluator.evaluateRecommender(RECOMMENDER)
print(" This is the MAP for FIrstLayer: {}".format( result_df.loc[10]["MAP"]))


