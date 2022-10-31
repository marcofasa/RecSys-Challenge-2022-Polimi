from Utils.Evaluator import EvaluatorHoldout
import numpy as np
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from datetime import datetime
import time
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import os
import Utils.Reader as Read

# Order the best map, with the same order with the name, topk and shrink
def order_MAP(MAP,topK,learning_rate,lambda1,lambda2):
    # Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP)):
        if (Best_MAP[index] < MAP):
            Best_MAP.insert(index, MAP)
            Best_Learning_Rate.insert(index, learning_rate)
            Best_topK.insert(index, topK)
            Best_Lambda1.insert(index,lambda1)
            Best_Lambda2.insert(index,lambda2)

            # If there was an adding and the length is >15, remove the last element
            if (len(Best_MAP) > max_length_best):
                del Best_MAP[-1]
                del Best_Learning_Rate[-1]
                del Best_Lambda1[-1]
                del Best_Lambda2[-1]
                del Best_topK[-1]

            return

    # If the array lenght is not 15, append the element
    if (len(Best_MAP) < max_length_best):
        Best_Learning_Rate.append(learning_rate)
        Best_Lambda1.append(lambda1)
        Best_Lambda2.append(lambda2)
        Best_topK.append(topK)

    return


# Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if (phase == "Training"):
        file = open("Testing_Results/SLIM_Best_Training.txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "     Learning rate: " + str(Best_Learning_Rate[index]) + "   topK: " + str(
                Best_topK[index]) +  "  lambda1 " + str(Best_Lambda1[index]) + "lamda2 " + str(Best_Lambda2[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Validation"):
        file = open("Testing_Results/SLIM_Best_Validation.txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) +"     Learning rate: " + str(Best_Learning_Rate[index]) + "   topK: " + str(
                Best_topK[index]) + "  lambda1 " + str(Best_Lambda1[index]) + "lamda2 " + str(
                Best_Lambda2[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Test"):
        file = open("Testing_Results/SLIM_Best_Test.txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "     Learning rate: " + str(Best_Learning_Rate[index]) + "   topK: " + str(
                Best_topK[index]) + "  lambda1 " + str(Best_Lambda1[index]) + "lamda2 " + str(
                Best_Lambda2[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return


def training_phase():
    for topK in x_tick_rnd_topK:
        for learning_rate in learning_rate_array:
            for lambda1 in x_tick_lamda1:
                for lambda2 in x_tick_lamda2:
            # x_tick.append("topk {}, shrink {}".format(topK, shrink))

                    SLIM_BPR.fit(epochs=num_epochs,lambda_i=lambda1,lambda_j=lambda2,learning_rate=learning_rate,topK=topK)

                    result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
                    print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with learning rate " + str(learning_rate) + " and topK " + str(topK) + " and lambda1 " + str(lambda1) + " and labda2  "+ str(lambda2))
                    order_MAP(result_df.loc[10]["MAP"], topK,learning_rate,lambda1,lambda2)

    save_data(phase="Training")
    return




# Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")

    for i in range(max_length_best):
        SLIM_BPR.fit(epochs=num_epochs, lambda_i=Best_Lambda1[i], lambda_j=Best_Lambda2[i],
                     learning_rate=Best_Learning_Rate[i], topK=Best_topK[i])

        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with learning rate " + str(
            Best_Learning_Rate[i]) + " and topK " + str(Best_topK[i]) + " and lambda1 " + str(
            Best_Lambda1[i]) + " and labda2  " + str(Best_Lambda2[i]))
        order_MAP(result_df.loc[10]["MAP"], Best_topK[i], Best_Learning_Rate[i], Best_Lambda1[i], Best_Lambda2[i])

    save_data(phase="Test")
    return


# Define the testing phase, based on the training and validation phase
def testing_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")


    for i in range(max_length_best):
        SLIM_BPR.fit(epochs=num_epochs, lambda_i=Best_Lambda1[i], lambda_j=Best_Lambda2[i], learning_rate=Best_Learning_Rate[i], topK=Best_topK[i])

        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with learning rate " + str(
            Best_Learning_Rate[i]) + " and topK " + str(Best_topK[i]) + " and lambda1 " + str(Best_Lambda1[i]) + " and labda2  " + str(Best_Lambda2[i]))
        order_MAP(result_df.loc[10]["MAP"], Best_topK[i], Best_Learning_Rate[i], Best_Lambda1[i], Best_Lambda2[i])

    save_data(phase="Test")
    return


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
# Keep the reference to the lambda1, lambda2 e learing_rate e  parameter sorted as the best MAP
Best_Learning_Rate = []
Best_Lambda1 = []
Best_Lambda2 = []
# Keep the reference to the topK paramter sorted as the bet MAP
Best_topK = []
# Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase
max_length_best = 40
# Variable for the num of parameter for topKin,lambda2 e lambda1 the test phase, the number of loops will be this number at the fourth
size_parameter = 5
# Start time
start_time = datetime.now().strftime("%D:  %H:%M:%S")
#Paramter for the number of epoch
num_epochs=300


evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

# knnn contenet filter recomennded none feature weighting
SLIM_BPR = SLIM_BPR_Cython(URM_train)
collaborative_None_MAP = []



# random search with the log uniform
from scipy.stats import loguniform

x_tick_rnd_topK = loguniform.rvs(10, 500, size=size_parameter).astype(int)
x_tick_rnd_topK.sort()
x_tick_rnd_topK = list(x_tick_rnd_topK)

x_tick_lamda1=[]
x_tick_lamda2=[]
learning_rate_array=[]
possible_values_learning_rate=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
#Randomly select the array for the parameter of lambda 1,2 and learning rate
for i in range(size_parameter):
    x_tick_lamda1.append(round(np.random.uniform(0.00001,0.5),5))
    x_tick_lamda2.append(round(np.random.uniform(0.00001,0.5),5))
    learning_rate_array.append(np.random.choice(possible_values_learning_rate))


training_phase()
validation_phase()
testing_phase()


#!!!!!!!!!!to compile cython run :  python CythonCompiler/compile_script.py Recommenders/SLIM/Cython/SLIM_BPR_Cython_Epoch.pyx build_ext --inplace!!!!!!!!!!!!
