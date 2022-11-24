import matplotlib.pyplot as plt

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample,split_train_in_two_percentage_global_sample_double

from Recommenders.Hybrid.FirstLayer import FirstLayer
from Recommenders.Hybrid.NewHybrid import NewHybrid
from Recommenders.Hybrid.NewHybrid1 import NewHybrid1
from Recommenders.Hybrid.P3_RP3 import P3_RP3
from Recommenders.Hybrid.Rankings import Rankings
from Utils import Reader
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Utils.Evaluator import EvaluatorHoldout
import numpy as np
import scipy.sparse as sps
import os


dirname = os.path.dirname(__file__)
ICM_path=  os.path.join(dirname, "data/data_ICM_type.csv")
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")
matrix_extended = os.path.join(dirname, "data/extended.csv")



#URM_train= Reader.read_train_csr(matrix_path)
#ICM_genres=Reader.read_ICM_type(matrix_path=ICM_path)

#URM_train,ICM_genres=Reader.get_URM_ICM_Type_Extended(matrix_path_URM=matrix_path,matrix_path_ICM_type=ICM_path)
#URM_train=Reader.read_train_csr(matrix_path=matrix_path)
import pandas as pd
import scipy.sparse as sp


URM_train = Reader.read_train_csr(matrix_path=matrix_path)
URM_extended=Reader.read_train_csr_extended(matrix_path=matrix_extended)
#URM_rewatches, ICM_genres= Reader.get_URM_ICM_Type_Extended(matrix_path_URM=rewatches_path, matrix_path_ICM_type=ICM_path)
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_extended, train_percentage=0.70)
#URM_extended,URM3,URM_train, URM_test = split_train_in_two_percentage_global_sample_double(URM_train, URM_extended,train_percentage=0.7)



profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
profile_length, profile_length.shape

block_size = int(len(profile_length)*0.1)
block_size
collaborative_recommender_class = {#"TopPop": TopPop,
                                   #"UserKNNCF": UserKNNCFRecommender,
                                   #"ItemKNNCF": ItemKNNCFRecommender,
                                   #"P3alpha": P3alphaRecommender,
                                   #"RP3beta": RP3betaRecommender,
                                   #"PureSVD": PureSVDRecommender,
                                   #"NMF": NMFRecommender,
                                   #"FunkSVD": MatrixFactorization_FunkSVD_Cython,
                                   #"IALS": IALSRecommender,
                                   #"SLIMBPR": SLIM_BPR_Cython,
                                    "SLIM_BPR_MASK":SLIM_BPR_Cython
                                   #"FirstLayer":FirstLayer,
                                    #"ItemKNNHybrid": ItemUserHybridKNNRecommender,
                                   #"NewHybrid": NewHybrid,
                                    #"NewHybrid1":NewHybrid1,
                                   #"P3_RP3": P3_RP3


                                   }
sorted_users = np.argsort(profile_length)



MAP_recommender_per_group= {}


content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender,
                             "ItemKNNCFCBF": ItemKNN_CFCBF_Hybrid_Recommender
                             }

recommender_object_dict = {}

for label, recommender_class in collaborative_recommender_class.items():

    recommender_object = recommender_class(URM_train)

    recommender_object.fit()
    recommender_object_dict[label] = recommender_object
'''
for label, recommender_class in content_recommender_class.items():
    recommender_object = recommender_class(URM_train, ICM_genres)
    recommender_object.fit()
    recommender_object_dict[label] = recommender_object
'''

cutoff = 10
for group_id in range(0,10):

    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    for label, recommender in recommender_object_dict.items():
        result_df, _  = evaluator_test.evaluateRecommender(recommender)
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]


_ = plt.figure(figsize=(16, 9))
for label, recommender in recommender_object_dict.items():
    results = MAP_recommender_per_group[label]
    plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
plt.ylabel('MAP')
plt.xlabel('User Group')
plt.legend()
plt.show()