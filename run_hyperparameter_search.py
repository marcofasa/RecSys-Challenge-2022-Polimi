#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
from Recommenders.Hybrid.FirstLayer import FirstLayer
from Recommenders.Recommender_import_list import *
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
import os as os
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
import traceback

import multiprocessing
from functools import partial
import numpy as np
import scipy.sparse as sps



from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
from Utils import Reader


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """





    output_folder_path = "result_experiments/"


    collaborative_algorithm_list = [
        #Random,
        #TopPop,
        #P3alphaRecommender,
        #RP3betaRecommender,
        #ItemKNNCFRecommender,
        #UserKNNCFRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        PureSVDRecommender,
        IALSRecommender,
        #NMFRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        #ItemUserHybridKNNRecommender,
        #FirstLayer,
    ]

    import os
    import pandas as pd
    import scipy.sparse as sp
    from Utils import Reader
    dirname = os.path.dirname(__file__)
    matrix_extended = os.path.join(dirname, "data/extended.csv")

    matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")
    ICM_path = os.path.join(dirname, "data/data_ICM_type.csv")
    ICM_path_length = os.path.join(dirname, "data/data_ICM_length.csv")

    replace = {0.01: -1, 0.5: 0, 0.8: 0, 0.2: 0}
    URM_train_normal = Reader.read_train_csr(matrix_path=matrix_path,values_to_replace={0:0.3})
    #URM_train_normal = Reader.load_URM(file_path=matrix_path)

    URM_train_last, URM_test = split_train_in_two_percentage_global_sample(URM_train_normal, 0.7)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_last, 0.7)

    #-----NEW URM------
    #URM_path2="../data/interactions_and_impressions_v4.csv"
    #URM_train,URM_train2 ,URM_test,URM_valid2=Reader.split_train_validation_double(URM_path2=URM_path2)
    #URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train2, 0.7)

    ########################
    #group_id=10 #con 2 sarebbe il 10 per cento ( cioè prende il 10 per cento dehgli user con meno intercation)
    profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
    block_size =10* int(len(profile_length) * 0.05)
    sorted_users = np.argsort(profile_length)
    start_pos = 0
    end_pos = min(block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]
    print("this is the numbers of user" + str(len(users_in_group)))
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    ignore_users = users_not_in_group

    from Utils.Evaluator import EvaluatorHoldout

    cutoff_list = [10]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 40
    n_random_starts = int(n_cases/3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       URM_train_last_test=URM_train_last,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path,
                                                       resume_from_saved = True,
                                                       similarity_type_list = ["cosine"],
                                                       parallelizeKNN = False)





    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)


'''
    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #


    ICM_dictionary={
        "ICM_TYPE": ICM_train
    }

    ################################################################################################
    ###### Content Baselines

    for ICM_name, ICM_object in ICM_dictionary.items():

        try:

            runHyperparameterSearch_Content(ItemKNNCBFRecommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)

        except Exception as e:

            print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
            traceback.print_exc()


        try:

            runHyperparameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)


        except Exception as e:

            print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
            traceback.print_exc()



'''

if __name__ == '__main__':


    read_data_split_and_search()
