#!/usr/bin/env python3

from tqdm import tqdm
from Utils import Reader
from Utils.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from datetime import datetime
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import multiprocessing
import numpy as np
# Order the best map, with the same order with the name, topk and shrink
def order_MAP(Best_MAP_phase,Best_shrink_phase,Best_similarity_phase,Best_TopK_Phase,MAP, shrink, topK, similarity):
    # Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP_phase)):
        if (Best_MAP_phase[index] < MAP):
            Best_MAP_phase.insert(index, MAP)
            Best_shrink_phase.insert(index, shrink)
            Best_TopK_Phase.insert(index, topK)
            Best_similarity_phase.insert(index, similarity)

            # If there was an adding and the length is >15, remove the last element
            if (len(Best_MAP_phase) > max_length_best):
                del Best_MAP_phase[-1]
                del Best_shrink_phase[-1]
                del Best_TopK_Phase[-1]
                del Best_similarity_phase[-1]

            return

    # If the array lenght is not 15, append the element
    if (len(Best_MAP_phase) < max_length_best):
        Best_MAP_phase.append(MAP)
        Best_similarity_phase.append(similarity)
        Best_shrink_phase.append(shrink)
        Best_TopK_Phase.append(topK)

    return


# Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase, Best_MAP_phase, Best_shrink_phase, Best_similarity_phase, Best_TopK_Phase):
    if (phase == "Training"):
        file = open("Testing_Results/CF_Best_Training_New_Value_Old_Model_" + str(multiprocessing.current_process())+".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_phase[index]) + "     Shrink: " + str(Best_shrink_phase[index]) + "   topK: " + str(
                Best_TopK_Phase[index]) + "   similarity:  " + str(Best_similarity_phase[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Validation"):
        file = open("Testing_Results/CF_Best_Validation_New_Value_Old_model_" + str(multiprocessing.current_process())+".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_phase[index]) + "     Shrink: " + str(Best_shrink_phase[index]) + "   topK: " + str(
                Best_TopK_Phase[index])+  "   similarity:  " + str(Best_similarity_phase[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Test"):
        file = open("Testing_Results/CF_Best_Test_New_Value_Old_Model_" + str(multiprocessing.current_process())+".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_phase[index]) + "     Shrink: " + str(Best_shrink_phase[index]) + "   topK: " + str(
                Best_TopK_Phase[index]) + "   similarity:  " + str(Best_similarity_phase[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return



def training_phase():
    for similarity in tqdm(similarities):
        for topK in tqdm(x_tick_rnd_topK, desc="training..."):
            for shrink in x_tick_rnd_shrink:
                # x_tick.append("topk {}, shrink {}".format(topK, shrink))
                collaborative_recommender_TF_IDF.fit(shrink=shrink, topK=topK, feature_weighting="TF-IDF", similarity=similarity)
                result_df, _ = evaluator_test.evaluateRecommender(collaborative_recommender_TF_IDF)
                collaborative_TF_IDF_MAP.append(result_df.loc[10]["MAP"])
                order_MAP(Best_MAP_phase=Best_MAP_training, Best_shrink_phase=Best_Shrink_training,
                          Best_similarity_phase=Best_similarities_training,
                          Best_TopK_Phase=Best_topK_training, MAP=result_df.loc[10]["MAP"], shrink=shrink, topK=topK,
                          similarity=similarity)

    save_data(phase="Training", Best_MAP_phase=Best_MAP_training,
              Best_similarity_phase=Best_similarities_training,
              Best_shrink_phase=Best_Shrink_training, Best_TopK_Phase=Best_topK_training)
    return


# Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")
    # knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = ItemKNNCFRecommender(URM_validation)
    evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    for i in tqdm(range(max_length_best)):
        content_recommender_TF_IDF.fit(shrink=Best_Shrink_training[i], topK=Best_topK_training[i],
                                       feature_weighting="TF-IDF", similarity=Best_similarities_training[i])
        result_df, _ = evaluator_test.evaluateRecommender(content_recommender_TF_IDF)
        order_MAP(Best_MAP_phase=Best_MAP_validation, Best_shrink_phase=Best_Shrink_validation,
                  Best_similarity_phase=Best_similarities_validation,
                  Best_TopK_Phase=Best_topK_validation, MAP=result_df.loc[10]["MAP"], shrink=Best_Shrink_training[i],
                  topK=Best_topK_training[i],
                  similarity=Best_similarities_training[i])

    save_data(phase="Validation", Best_MAP_phase=Best_MAP_validation,
              Best_similarity_phase=Best_similarities_validation,
              Best_shrink_phase=Best_Shrink_validation, Best_TopK_Phase=Best_topK_validation)
    return

# Define the testing phase, based on the training and validation phase
def testing_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")
    # knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = ItemKNNCFRecommender(URM_test)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    for i in tqdm(range(max_length_best), desc="testing phase..."):

            content_recommender_TF_IDF.fit(shrink=Best_Shrink_validation[i], topK=Best_topK_validation[i], feature_weighting="TF-IDF", similarity=Best_similarities_validation[i])
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_TF_IDF)
            order_MAP(Best_MAP_phase=Best_MAP_testing, Best_shrink_phase=Best_Shrink_testing,
                      Best_similarity_phase=Best_similarities_testing,
                      Best_TopK_Phase=Best_topK_testing, MAP=result_df.loc[10]["MAP"],
                      shrink=Best_Shrink_validation[i], topK=Best_topK_validation[i],
                      similarity=Best_similarities_validation[i])

    save_data(phase="Test", Best_MAP_phase=Best_MAP_testing,
              Best_similarity_phase=Best_similarities_testing,
              Best_shrink_phase=Best_Shrink_testing, Best_TopK_Phase=Best_topK_testing)
    return

# Keep the reference to the BestMAP in each phase
Best_MAP_training = []
Best_MAP_validation = []
Best_MAP_testing = []
# Keep the reference to the shrink parameter sorted as the best MAP
Best_Shrink_training = []
Best_Shrink_validation = []
Best_Shrink_testing = []
# Keep the reference to the topK paramter sorted as the bet MAP
Best_topK_training = []
Best_topK_validation = []
Best_topK_testing = []
#Keep the refernce to the best similarities
Best_similarities_training=[]
Best_similarities_validation=[]
Best_similarities_testing=[]

# Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase, MUST BE GREATER OR EQUAL THAN THE SIZE PARAMETER SQURED
max_length_best = 50

# Variable for the num of parameter for shrink and topKin the test phase, the number of loops will be this number squared
size_parameter = 10
# Start timeb
start_time = datetime.now().strftime("%D:  %H:%M:%S")
#similarities to test
similarities=["cosine"]


# random search with the log uniform
from scipy.stats import loguniform

#Let's try rand int
#x_tick_rnd_topK=np.random.randint(100,size=10)
x_tick_rnd_topK = loguniform.rvs(10, 500, size=size_parameter).astype(int)
x_tick_rnd_topK.sort()
x_tick_rnd_topK = list(x_tick_rnd_topK)

x_tick_rnd_shrink = loguniform.rvs(10, 500, size=size_parameter).astype(int)
#x_tick_rnd_shrink=np.random.randint(10,size=10)
x_tick_rnd_shrink.sort()
x_tick_rnd_shrink = list(x_tick_rnd_topK)


import os
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")

URM_all = Reader.read_train_csr(matrix_path=matrix_path)

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


# knnn contenet filter recomennded TF_IDF feature weighting
collaborative_recommender_TF_IDF = ItemKNNCFRecommender(URM_train)
collaborative_TF_IDF_MAP = []




def start_parameter_tuning(x):


    global x_tick_rnd_topK
    global x_tick_rnd_shrink
    x_tick_rnd_topK = loguniform.rvs(x, 580, size=size_parameter).astype(int)
    x_tick_rnd_topK.sort()
    x_tick_rnd_topK = list(x_tick_rnd_topK)

    x_tick_rnd_shrink = loguniform.rvs(x, 500, size=size_parameter).astype(int)
    # x_tick_rnd_shrink=np.random.randint(10,size=10)
    x_tick_rnd_shrink.sort()
    x_tick_rnd_shrink = list(x_tick_rnd_shrink)

    training_phase()
    validation_phase()
    testing_phase()


pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)

n_thread=[350,380,300,500,450,420]


pool.map(start_parameter_tuning, n_thread)




#TODO: try other similarity!!