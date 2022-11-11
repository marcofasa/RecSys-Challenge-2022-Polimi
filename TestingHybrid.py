from Utils.Evaluator import EvaluatorHoldout
import numpy as np
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from datetime import datetime
import time
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import os
import Utils.Reader as Read
from scipy.stats import loguniform
import pandas as pd
import scipy.sparse as sp
import os, multiprocessing
from tqdm import tqdm

# Order the best map, with the same order with the name, topk and shrink

# Order the best map, with the same order with the name, topk and shrink
def order_MAP(Best_MAP_phase,Best_learning_rate_phase,Best_lambda1_phase,Best_lambda2_phase,Best_TopK_Phase,Best_alpha_phase,MAP,
              lambda1,lambda2,learning_rate, topK, alpha):
    # Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP_phase)):
        if (Best_MAP_phase[index] < MAP):
            Best_MAP_phase.insert(index, MAP)
            Best_learning_rate_phase.insert(index, learning_rate)
            Best_TopK_Phase.insert(index, topK)
            Best_lambda1_phase.insert(index, lambda1)
            Best_lambda2_phase.insert(index,lambda2)
            Best_alpha_phase.insert(index,alpha)

            # If there was an adding and the length is >15, remove the last element
            if (len(Best_MAP_phase) > max_length_best):
                del Best_MAP_phase[-1]
                del Best_learning_rate_phase[-1]
                del Best_TopK_Phase[-1]
                del Best_lambda1_phase[-1]
                del Best_lambda2_phase[-1]
                del Best_alpha_phase[-1]
            return

    # If the array lenght is not 15, append the element
    if (len(Best_MAP_phase) < max_length_best):
        Best_MAP_phase.append(MAP)
        Best_learning_rate_phase.append(learning_rate)
        Best_lambda1_phase.append(lambda1)
        Best_TopK_Phase.append(topK)
        Best_lambda2_phase.append(lambda2)
        Best_alpha_phase.append(alpha)

    return



# Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if (phase == "Training"):
        file = open("Testing_Results/ItemUser_Best_Training_"+ str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(len(Best_MAP_training)):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_training[index]) + "     Learning rate: " +
                       str(Best_topK_CF_Rate_training[index]) + "   topK: " + str(
                Best_topK_training[index]) + "  topk_CF " + str(Best_shrink_CF_training[index]) + "shrink " + str(Best_shrink_training[index]) +
                       " alpha= " + str(Best_alpha_training[index])  + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Validation"):
        file = open("Testing_Results/ItemUser_SLIM_Best_Validation_"+ str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_training[index]) + "     topkCF  rate: " +
                       str(Best_topK_CF_Rate_validation[index]) + "   topK: " + str(
                Best_topK_training[index]) + "  topk_CF " + str(Best_shrink_CF_validation[index]) + "shrink " + str(
                Best_shrink_training[index]) +
                       " alpha= " + str(Best_alpha_training[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Test"):
        file = open("Testing_Results/ITEMKNN_SLIM_Best_Test_" + str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_testing[index]) + "     Learning rate: " + str(Best_Learning_Rate_testing[index]) + "   topK: " + str(
              #  Best_topK_testing[index]) + "  lambda1 " + str(Best_Lambda1_testing[index]) + "lamda2 " + str(
                Best_Lambda2_testing[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return


def training_phase():

    for alpha_element in alpha:
        for topK_CF in tqdm(x_tick_rnd_topK_CF):
            for topK in x_tick_rnd_topK:
                for shrink_ele in tqdm(shrink , desc="second cicle"):
                    for shrink_CF_ele in shrink_CF:

                    # x_tick.append("topk {}, shrink {}".format(topK, shrink))

                        SLIM_BPR.fit(topK=topK_CF,topK_CF=topK,alpha=alpha_element,shrink_CF=shrink_CF_ele,shrink=shrink_ele)

                        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
                        print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with shrinkCF rate " + str(shrink_CF_ele) +
                              " and topK " + str(topK_CF) + " and shrinkCF " + str(shrink_CF) + " and topk User  "+ str(topK))
                        order_MAP(MAP=result_df.loc[10]["MAP"], topK=topK, learning_rate=topK_CF, lambda1=shrink_ele, lambda2=shrink_CF_ele,
                                  Best_learning_rate_phase=Best_topK_CF_Rate_training, Best_MAP_phase=Best_MAP_training, Best_TopK_Phase=Best_topK_training,
                                  Best_lambda1_phase=Best_shrink_training, Best_lambda2_phase=Best_shrink_CF_training, Best_alpha_phase=Best_alpha_training,
                                   alpha=alpha_element)
                        save_data(phase="Training")

    return




# Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    SLIM_BPR = ItemUserHybridKNNRecommender(URM_validation)
    for i in tqdm(range(max_length_best)):
        # x_tick.append("topk {}, shrink {}".format(topK, shrink))

        SLIM_BPR.fit(topK=Best_topK_training[i], topK_CF=Best_topK_CF_Rate_training[i], alpha=Best_alpha_training[i],
                     shrink_CF=Best_shrink_CF_training[i], shrink=Best_shrink_training[i])

        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with shrinkCF rate " + str(Best_shrink_CF_training[i]) +
              " and topK " + str(Best_topK_CF_Rate_validation[i]) + " and shrinkCF " + str(Best_shrink_training[i])
              + " and topk User  " + str(Best_topK_training[i]))
        order_MAP(MAP=result_df.loc[10]["MAP"], topK=Best_topK_training[i], learning_rate=Best_topK_CF_Rate_training[i],
                  lambda1=Best_shrink_training[i], lambda2=Best_shrink_CF_training[i],
                  Best_learning_rate_phase=Best_topK_CF_Rate_validation, Best_MAP_phase=Best_MAP_validation,
                  Best_TopK_Phase=Best_topK_validation,
                  Best_lambda1_phase=Best_shrink_validation, Best_lambda2_phase=Best_shrink_CF_validation,
                  Best_alpha_phase=Best_alpha_validation,
                  alpha=Best_alpha_training[i])

    save_data(phase="Validation")
    return


def start_parameter_tuning(x):
    global x_tick_rnd_topK
    global shrink_CF
    global shrink
    global  x_tick_rnd_topK_CF

    x_tick_rnd_topK = loguniform.rvs(100, 900, size=size_parameter).astype(int)
    x_tick_rnd_topK.sort()
    x_tick_rnd_topK = list(x_tick_rnd_topK)

    shrink_CF=loguniform.rvs(100, 900, size=size_parameter).astype(int)
    shrink_CF.sort()
    shrink_CF = list(shrink_CF)

    shrink = loguniform.rvs(100, 9000, size=size_parameter).astype(int)
    shrink.sort()
    shrink = list(shrink)

    x_tick_rnd_topK_CF=loguniform.rvs(100, 1000, size=size_parameter).astype(int)
    x_tick_rnd_topK_CF.sort()
    x_tick_rnd_topK_CF = list(x_tick_rnd_topK_CF)




    training_phase()
    validation_phase()
    #testing_phase()


from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")

rewatches_path=os.path.join(dirname, "data/rewatches.csv")

URM_train=pd.read_csv(rewatches_path,sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: int},
                            engine='python')

columns=["UserID","ItemID","data"]
URM_train.columns=columns
print(URM_train)
URM_train = sp.coo_matrix((URM_train[columns[2]].values,
                         (URM_train[columns[0]].values, URM_train[columns[1]].values)
                         ))

URM_rewatches=URM_train.tocsr()
#URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
URM=Read.read_train_csr(matrix_path=matrix_path)
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)



"""
seed = 1234

(user_ids_training, user_ids_test,
 item_ids_training, item_ids_test,
 ratings_training, ratings_test) = train_test_split(URM_normal[0],
                                                    URM_normal[1],
                                                    URM_normal[2],
                                                    test_size=0.8,
                                                    shuffle=True,
                                                    random_state=seed)

URM_train = sp.csr_matrix(ratings_training, (user_ids_training, item_ids_training))
URM_normal = sp.csr_matrix(URM_normal[2], (user_ids_training, item_ids_training))
"""

#URM_normal, URM_test_normal= split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
#URM_normal, URM_validation_normal = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
# Keep the reference to the BestMAP in each phase
Best_MAP_training = []
Best_MAP_validation = []
Best_MAP_testing=[]



Best_shrink_CF_training=[]
Best_shrink_CF_validation=[]
# Keep the reference to the lambda1, lambda2 e learing_rate e  parameter sorted as the best MAP
Best_topK_CF_Rate_training = []
Best_topK_CF_Rate_validation=[]
Best_Learning_Rate_testing=[]
Best_shrink_training = []
Best_shrink_validation=[]
Best_alpha_training = []
Best_alpha_validation=[]
Best_Lambda2_testing=[]
# Keep the reference to the topK paramter sorted as the bet MAP
Best_topK_training = []
Best_topK_validation=[]
Best_topK_testing=[]
# Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase
max_length_best = 40
# Variable for the num of parameter for topKin,lambda2 e lambda1 the test phase, the number of loops will be this number at the fourth
size_parameter = 9
# Start time
start_time = datetime.now().strftime("%D:  %H:%M:%S")
#Paramter for the number of epoch
alpha=[0.5,0.6,0.7]
shrink_CF = []
shrink = []
x_tick_rnd_topK_CF = []
x_tick_rnd_topK=[]

evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[10])

from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
# knnn contenet filter recomennded none feature weighting
SLIM_BPR = ItemUserHybridKNNRecommender(URM_train)
collaborative_None_MAP = []

start_parameter_tuning(2)

#!!!!!!!!!!to compile cython run :  python CythonCompiler/compile_script.py Recommenders/SLIM/Cython/SLIM_BPR_Cython_Epoch.pyx build_ext --inplace!!!!!!!!!!!!
