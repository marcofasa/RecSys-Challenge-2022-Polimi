import scipy.sparse as sps
import pandas as pd
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from datetime import datetime
import time
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import os

# Order the best map, with the same order with the name, topk and shrink
def order_MAP(name, MAP, shrink, topK):
    # Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP)):
        if (Best_MAP[index] < MAP):
            Best_MAP.insert(index, MAP)
            Model_type.insert(index, name)
            Best_Shrink.insert(index, shrink)
            Best_topK.insert(index, topK)

            # If there was an adding and the length is >15, remove the last element
            if (len(Best_MAP) > max_length_best):
                del Best_MAP[-1]
                del Model_type[-1]
                del Best_Shrink[-1]
                del Best_topK[-1]

            return

    # If the array lenght is not 15, append the element
    if (len(Best_MAP) < max_length_best):
        Best_MAP.append(MAP)
        Model_type.append(name)
        Best_Shrink.append(shrink)
        Best_topK.append(topK)

    return


# Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if (phase == "Training"):
        file = open("Testing_Results/CF_Best_Training.txt", "w+")
        for index in range(15):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(
                Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(
                Best_topK[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Validation"):
        file = open("Testing_Results/CF_Best_Validation.txt", "w+")
        for index in range(15):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(
                Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(
                Best_topK[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Test"):
        file = open("Testing_Results/CF_Best_Test.txt", "w+")
        for index in range(15):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(
                Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(
                Best_topK[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return


def training_phase():
    for topK in x_tick_rnd_topK:

            # x_tick.append("topk {}, shrink {}".format(topK, shrink))

            SLIM_BPR.fit(epochs=3)

            result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
            print("This is the None MAP " + str(result_df.loc[10]["MAP"]) + " with shrink term " + str(
                shrink) + " and topK " + str(topK))
            order_MAP("None", result_df.loc[10]["MAP"], shrink, topK)

    save_data(phase="Training")
    return

import Utils.Reader as Read

import os
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")

URM_all=Read.read_train_csr(matrix_path)
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.60)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)

# Keep the reference to the BestMAP in each phase
Best_MAP = []
# Keep the referenceto the Model Type(No weigh, BM25, TF-IDF) sorted as the best MAP
Model_type = []
# Keep the reference to the shrink parameter sorted as the best MAP
Best_Shrink = []
# Keep the reference to the topK paramter sorted as the bet MAP
Best_topK = []
# Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase
max_length_best = 15
# Variable for the num of parameter for shrink and topKin the test phase, the number of loops will be this number squared
size_parameter = 8
# Start time
start_time = datetime.now().strftime("%D:  %H:%M:%S")

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

# knnn contenet filter recomennded none feature weighting
SLIM_BPR = SLIM_BPR_Cython(URM_train)
collaborative_None_MAP = []



# random search with the log uniform
from scipy.stats import loguniform

x_tick_rnd_topK = loguniform.rvs(10, 500, size=size_parameter).astype(int)
x_tick_rnd_topK.sort()
x_tick_rnd_topK = list(x_tick_rnd_topK)





#!!!!!!!!!!to compile cython run :  python CythonCompiler/compile_script.py Recommenders/SLIM/Cython/SLIM_BPR_Cython_Epoch.pyx build_ext --inplace!!!!!!!!!!!!
