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
def order_MAP(Best_MAP_phase,Best_learning_rate_phase,Best_lambda1_phase,Best_lambda2_phase,Best_TopK_Phase,MAP,
              lambda1,lambda2,learning_rate, topK):
    # Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP_phase)):
        if (Best_MAP_phase[index] < MAP):
            Best_MAP_phase.insert(index, MAP)
            Best_learning_rate_phase.insert(index, learning_rate)
            Best_TopK_Phase.insert(index, topK)
            Best_lambda1_phase.insert(index, lambda1)
            Best_lambda2_phase.insert(index,lambda2)

            # If there was an adding and the length is >15, remove the last element
            if (len(Best_MAP_phase) > max_length_best):
                del Best_MAP_phase[-1]
                del Best_learning_rate_phase[-1]
                del Best_TopK_Phase[-1]
                del Best_lambda1_phase[-1]
                del Best_lambda2_phase[-1]
            return

    # If the array lenght is not 15, append the element
    if (len(Best_MAP_phase) < max_length_best):
        Best_MAP_phase.append(MAP)
        Best_learning_rate_phase.append(learning_rate)
        Best_lambda1_phase.append(lambda1)
        Best_TopK_Phase.append(topK)
        Best_lambda2_phase.append(lambda2)

    return



# Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if (phase == "Training"):
        file = open("Testing_Results/SLIM_Best_Training_"+ str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(len(Best_MAP_testing)):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_training[index]) + "     Learning rate: " +
                       str(Best_Learning_Rate_training[index]) + "   topK: " + str(
                Best_topK_training[index]) +  "  lambda1 " + str(Best_Lambda1_training[index]) + "lamda2 " + str(Best_Lambda2_training[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Validation"):
        file = open("Testing_Results/SLIM_Best_Validation_"+ str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_validation[index]) +"     Learning rate: " + str(Best_Learning_Rate_validation[index]) + "   topK: " + str(
                Best_topK_validation[index]) + "  lambda1 " + str(Best_lambda1_validation[index]) + "lamda2 " + str(
                Best_lambda2_validation[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif (phase == "Test"):
        file = open("Testing_Results/SLIM_Best_Test_" + str(multiprocessing.current_process) + ".txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP_testing[index]) + "     Learning rate: " + str(Best_Learning_Rate_testing[index]) + "   topK: " + str(
                Best_topK_testing[index]) + "  lambda1 " + str(Best_Lambda1_testing[index]) + "lamda2 " + str(
                Best_Lambda2_testing[index]) + "\n")
        file.write("\nStarted at:  " + str(start_time) + "\nFinished at (Date-Time):   " + str(
            datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return


def training_phase():
    for topK in tqdm(x_tick_rnd_topK):
        for learning_rate in tqdm(learning_rate_array ,desc="second cicle"):
            for i in tqdm(range(size_parameter),desc="first circle"):

            # x_tick.append("topk {}, shrink {}".format(topK, shrink))

                SLIM_BPR.fit(epochs=num_epochs,lambda_i=x_tick_lamda1[i],lambda_j=x_tick_lamda2[i],learning_rate=learning_rate,topK=topK)

                result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
                print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with learning rate " + str(learning_rate) + " and topK " + str(topK) + " and lambda1 " + str(x_tick_lamda1[i]) + " and labda2  "+ str(x_tick_lamda2[i]))
                order_MAP(MAP=result_df.loc[10]["MAP"], topK=topK,learning_rate=learning_rate,lambda1=x_tick_lamda1[i],lambda2=x_tick_lamda2[i],
                          Best_learning_rate_phase=Best_Learning_Rate_training, Best_MAP_phase=Best_MAP_training, Best_TopK_Phase=Best_topK_training,
                          Best_lambda1_phase=Best_Lambda1_training,Best_lambda2_phase=Best_Lambda2_training)
    save_data(phase="Training")

    return




# Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")
    #evaluator_test = EvaluatorHoldout(URM_validation_normal, cutoff_list=[10])
    SLIM_BPR = SLIM_BPR_Cython(URM_validation)
    for i in tqdm(range(max_length_best)):
        SLIM_BPR.fit(epochs=num_epochs, lambda_i=Best_Lambda1_training[i], lambda_j=Best_Lambda2_training[i],
                     learning_rate=Best_Learning_Rate_training[i], topK=Best_topK_training[i])


        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)
        print("This is the MAP " + str(result_df.loc[10]["MAP"]) + " with learning rate " + str(
            Best_Learning_Rate_training[i]) + " and topK " + str(Best_topK_training[i]) + " and lambda1 " + str(
            Best_Lambda1_training[i]) + " and labda2  " + str(Best_Lambda2_training[i]))
        order_MAP(MAP=result_df.loc[10]["MAP"], topK=Best_topK_training[i], learning_rate=Best_Learning_Rate_training[i], lambda1=Best_Lambda1_training[i],
                  lambda2=Best_Lambda2_training[i],
                  Best_learning_rate_phase=Best_Learning_Rate_validation, Best_MAP_phase=Best_MAP_validation,
                  Best_TopK_Phase=Best_topK_validation,
                  Best_lambda1_phase=Best_lambda1_validation, Best_lambda2_phase=Best_lambda2_validation)

    save_data(phase="Validation")
    return


# Define the testing phase, based on the training and validation phase
def testing_phase():
    global start_time
    start_time = datetime.now().strftime("%D:  %H:%M:%S")
    #evaluator_test = EvaluatorHoldout(URM_test_normal, cutoff_list=[10])
    SLIM_BPR = SLIM_BPR_Cython(URM_test)
    for i in tqdm(range(max_length_best)):
        SLIM_BPR.fit(epochs=num_epochs, lambda_i=Best_lambda1_validation[i], lambda_j=Best_lambda2_validation[i], learning_rate=Best_Learning_Rate_validation[i],
                     topK=Best_topK_validation[i])

        result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR)

        order_MAP(MAP=result_df.loc[10]["MAP"], topK=Best_topK_validation[i],
                  learning_rate=Best_Learning_Rate_validation[i], lambda1=Best_lambda1_validation[i],
                  lambda2=Best_lambda2_validation[i],
                  Best_learning_rate_phase=Best_Learning_Rate_testing, Best_MAP_phase=Best_MAP_testing,
                  Best_TopK_Phase=Best_topK_testing,
                  Best_lambda1_phase=Best_Lambda1_testing, Best_lambda2_phase=Best_Lambda2_testing)

    save_data(phase="Test")
    return


def start_parameter_tuning(x):
    global x_tick_rnd_topK
    global x_tick_lamda1
    global x_tick_lamda2
    global  learning_rate_array

    x_tick_rnd_topK = loguniform.rvs(50, 500, size=size_parameter).astype(int)
    x_tick_rnd_topK.sort()
    x_tick_rnd_topK = list(x_tick_rnd_topK)


    possible_values_learning_rate = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # Randomly select the array for the parameter of lambda 1,2 and learning rate
    for i in range(size_parameter):
        x_tick_lamda1.append(round(np.random.uniform(0.00001, 0.5), 5))
        x_tick_lamda2.append(round(np.random.uniform(0.00001, 0.5), 5))
        learning_rate_array.append(np.random.choice(possible_values_learning_rate))

    training_phase()
    validation_phase()
    testing_phase()


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

URM_train.tocsr()
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
URM_normal=Read.read_train_csr(matrix_path=matrix_path)



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

# Keep the reference to the lambda1, lambda2 e learing_rate e  parameter sorted as the best MAP
Best_Learning_Rate_training = []
Best_Learning_Rate_validation=[]
Best_Learning_Rate_testing=[]
Best_Lambda1_training = []
Best_lambda1_validation=[]
Best_Lambda1_testing=[]
Best_Lambda2_training = []
Best_lambda2_validation=[]
Best_Lambda2_testing=[]
# Keep the reference to the topK paramter sorted as the bet MAP
Best_topK_training = []
Best_topK_validation=[]
Best_topK_testing=[]
# Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase
max_length_best = 10
# Variable for the num of parameter for topKin,lambda2 e lambda1 the test phase, the number of loops will be this number at the fourth
size_parameter = 3
# Start time
start_time = datetime.now().strftime("%D:  %H:%M:%S")
#Paramter for the number of epoch
num_epochs=300
x_tick_lamda1 = []
x_tick_lamda2 = []
learning_rate_array = []
x_tick_rnd_topK=[]

evaluator_test = EvaluatorHoldout(URM_normal, cutoff_list=[10])

# knnn contenet filter recomennded none feature weighting
SLIM_BPR = SLIM_BPR_Cython(URM_train)
collaborative_None_MAP = []

pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
pool.map(start_parameter_tuning, range(12))


#!!!!!!!!!!to compile cython run :  python CythonCompiler/compile_script.py Recommenders/SLIM/Cython/SLIM_BPR_Cython_Epoch.pyx build_ext --inplace!!!!!!!!!!!!
