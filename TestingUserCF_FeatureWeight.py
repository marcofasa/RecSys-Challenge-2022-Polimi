from Utils.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from datetime import datetime
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
import os
from tqdm import tqdm





#Order the best map, with the same order with the name, topk and shrink
def order_MAP(name,MAP,shrink,topK):
    #Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP)):
        if(Best_MAP[index]<MAP):
            Best_MAP.insert(index,MAP)
            Model_type.insert(index,name)
            Best_Shrink.insert(index,shrink)
            Best_topK.insert(index,topK)

            #If there was an adding and the length is >15 
            if(len(Best_MAP)>max_length_best):
                del Best_MAP[-1]
                del Model_type[-1]
                del Best_Shrink[-1]
                del Best_topK[-1]

            return

    #If the array lenght is not max_lenght_best, append the element
    if(len(Best_MAP)<max_length_best):
        Best_MAP.append(MAP)
        Model_type.append(name)
        Best_Shrink.append(shrink)
        Best_topK.append(topK)

    return


#Write the best MAP with their name anda parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if(phase=="Training"):
        file= open("Testing_Results/CBFR_Best_Training.txt", "w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif(phase=="Validation"):
        file= open("Testing_Results/CBFR_Best_Validation.txt","w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))        
        file.close()
        return
    elif(phase=="Test"):
        file= open("Testing_Results/CBFR_Best_Test.txt","w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return

    

def training_phase():
    for topK in tqdm(x_tick_rnd_topK):
        for shrink in x_tick_rnd_shrink:

            #x_tick.append("topk {}, shrink {}".format(topK, shrink))

            content_recommender_none.fit(shrink=shrink, topK=topK)
            content_recommender_BM25.fit(shrink=shrink, topK=topK, feature_weighting="BM25")
            content_recommender_TF_IDF.fit(shrink=shrink,topK=topK, feature_weighting="TF-IDF")


            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_none)
            content_None_MAP.append(result_df.loc[10]["MAP"])
            print("This is the None MAP " + str(result_df.loc[10]["MAP"]) + " with shrink term " + str(shrink) + " and topK " + str(topK))
            order_MAP("None",result_df.loc[10]["MAP"],shrink,topK)

            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_BM25)
            content_BM25_MAP.append(result_df.loc[10]["MAP"])
            print("This is the BM25 MAP " + str(result_df.loc[10]["MAP"])+ " with shrink term " + str(shrink) + " and topK " + str(topK))
            order_MAP("BM25",result_df.loc[10]["MAP"],shrink,topK)

            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_TF_IDF)
            content_TF_IDF_MAP.append(result_df.loc[10]["MAP"])
            print("This is the IDF MAP " + str(result_df.loc[10]["MAP"])+ " with shrink term " + str(shrink) + " and topK " + str(topK))
            order_MAP("TF-IDF",result_df.loc[10]["MAP"],shrink,topK)

    save_data(phase="Training")
    return

#pyplot.plot(x_tick, content_None_MAP, label="None FW")
#pyplot.plot(x_tick, content_BM25_MAP, label="BM25")
#pyplot.plot(x_tick, content_TF_IDF_MAP, label="IDF")   

#pyplot.ylabel('Similarity')
#pyplot.xlabel('Sorted values')
#pyplot.legend()
#pyplot.show()


#Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    global start_time
    start_time=datetime.now().strftime("%D:  %H:%M:%S")

    evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    #knnn contenet filter recomennded none feature weighting
    content_recommender_none = UserKNNCFRecommender(URM_validation)
    
    #knnn contenet filter recomennded BM25 feature weighting
    content_recommender_BM25 = UserKNNCFRecommender(URM_validation)
    
    #knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = UserKNNCFRecommender(URM_validation)
    
    for i in range(max_length_best):
        if(Model_type[i]=="None"):
            content_recommender_none.fit(shrink=Best_Shrink[i], topK= Best_topK[i])
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_none)
            order_MAP("None",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])
        elif(Model_type[i]=="BM25"):
            content_recommender_BM25.fit(shrink=Best_Shrink[i], topK=Best_topK[i],feature_weighting="BM25")
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_BM25)
            order_MAP("BM25",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])
        elif(Model_type[i]=="TF-IDF"):
            content_recommender_TF_IDF.fit(shrink=Best_Shrink[i],topK= Best_topK[i], feature_weighting="TF-IDF")
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_TF_IDF)
            order_MAP("TF-IDF",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])

    save_data(phase="Validation")
    return





#Define the testing phase, based on the training and validation phase
def testing_phase():
    global start_time
    start_time=datetime.now().strftime("%D:  %H:%M:%S")

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])


    #knnn contenet filter recomennded none feature weighting
    content_recommender_none = UserKNNCFRecommender(URM_test)
    

    #knnn contenet filter recomennded BM25 feature weighting
    content_recommender_BM25 = UserKNNCFRecommender(URM_test)
    

    #knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = UserKNNCFRecommender(URM_test)
    

    for i in range(max_length_best):
        if(Model_type[i]=="None"):
            content_recommender_none.fit(shrink=Best_Shrink[i], topK= Best_topK[i])
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_none)
            order_MAP("None",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])
        elif(Model_type[i]=="BM25"):
            content_recommender_BM25.fit(shrink=Best_Shrink[i], topK=Best_topK[i],feature_weighting="BM25")
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_BM25)
            order_MAP("BM25",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])
        elif(Model_type[i]=="TF-IDF"):
            content_recommender_TF_IDF.fit(shrink=Best_Shrink[i],topK= Best_topK[i], feature_weighting="TF-IDF")
            result_df, _ = evaluator_test.evaluateRecommender(content_recommender_TF_IDF)
            order_MAP("TF-IDF",result_df.loc[10]["MAP"],Best_Shrink[i], Best_topK[i])

    save_data(phase="Test")
    return

#Declaring the URM and splitting the dataset
import pandas as pd
from Utils import Reader

dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)



#Keep the reference to the BestMAP in each phase
Best_MAP=[]
#Keep the referenceto the Model Type(No weigh, BM25, TF-IDF) sorted as the best MAP
Model_type=[]
#Keep the reference to the shrink parameter sorted as the best MAP
Best_Shrink=[]
#Keep the reference to the topK paramter sorted as the bet MAP
Best_topK=[]
#Parameter that declare how many of the best parameter store in the array
max_length_best=30
#Variable for the num of parameter for shrink and topKin the test phase
size_parameter=30
#Start time
start_time=datetime.now().strftime("%D:  %H:%M:%S")

evaluator_test = EvaluatorHoldout(URM_train, cutoff_list=[10])


#knnn contenet filter recomennded none feature weighting
content_recommender_none = UserKNNCFRecommender(URM_train)
content_None_MAP = []

#knnn contenet filter recomennded BM25 feature weighting
content_recommender_BM25 = UserKNNCFRecommender(URM_train)
content_BM25_MAP = []

#knnn contenet filter recomennded TF_IDF feature weighting
content_recommender_TF_IDF = UserKNNCFRecommender(URM_train)
content_TF_IDF_MAP = []


#random search with the log uniform
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


training_phase()
validation_phase()
testing_phase()
#TODO: Implement a loop for each similarities(cosine, jaccard...) 