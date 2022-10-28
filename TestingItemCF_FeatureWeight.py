import pandas as pd
import scipy.sparse as sps
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from datetime import datetime
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

def load_URM():

    URM_path = "/home/vittorio/Scrivania/Politecnico/RecSys/RecSys_DEPRECATED/Dataset/interactions_and_impressions.csv"
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                    sep=",",
                                    header=None,
                                   # dtype={0:int, 1:int, 2:str,4:int},
                                    engine='python')  # its a way to store the data, they are sepatated by sep

    URM_all_dataframe.columns = ["UserID", "ItemID", "Impression_list", "Data"]
    print(URM_all_dataframe.head(n=10))
    userID_unique = URM_all_dataframe["UserID"].unique()
    itemID_unique = URM_all_dataframe["ItemID"].unique()
    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    n_interactions = len(URM_all_dataframe)

    print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemID_unique), max(userID_unique)))

    mapped_id, original_id = pd.factorize(
    URM_all_dataframe["UserID"].unique())  # take all the unique id and delete the empty profile
    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(URM_all_dataframe["ItemID"].unique())

    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    URM_all_dataframe["UserID"] = URM_all_dataframe["UserID"].map(user_original_ID_to_index)
    URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)
    print(URM_all_dataframe.head(n=10))


    URM_all_dataframe["Data"][0]=0
    URM_all = sps.coo_matrix((URM_all_dataframe["Data"].values,
                              (URM_all_dataframe["UserID"].values,
                               URM_all_dataframe["ItemID"].values)))  # fast format for constructing sparse matrices

    print(URM_all)

    return URM_all


#Order the best map, with the same order with the name, topk and shrink
def order_MAP(name,MAP,shrink,topK):
    #Check if the MAP is less than the one alredy stored and insert it at that index
    for index in range(len(Best_MAP)):
        if(Best_MAP[index]<MAP):
            Best_MAP.insert(index,MAP)
            Model_type.insert(index,name)
            Best_Shrink.insert(index,shrink)
            Best_topK.insert(index,topK)

            #If there was an adding and the length is >15, remove the last element
            if(len(Best_MAP)>max_length_best):
                del Best_MAP[-1]
                del Model_type[-1]
                del Best_Shrink[-1]
                del Best_topK[-1]

            return

    #If the array lenght is not 15, append the element
    if(len(Best_MAP)<max_length_best):
        Best_MAP.append(MAP)
        Model_type.append(name)
        Best_Shrink.append(shrink)
        Best_topK.append(topK)

    return


#Write the best MAP with their name and parameters in a textfile in the directory Testing_Results
def save_data(phase):
    if(phase=="Training"):
        file= open("Testing_Results/CF_Best_Training.txt","w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif(phase=="Validation"):
        file= open("Testing_Results/CF_Best_Validation.txt","w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return
    elif(phase=="Test"):
        file= open("Testing_Results/CF_Best_Test.txt","w+")
        for index in range(max_length_best):
            file.write(str(index) + ".  MAP: " + str(Best_MAP[index]) + "    Name: " + str(Model_type[index]) + "     Shrink: " + str(Best_Shrink[index]) + "   topK: " + str(Best_topK[index]) + "\n")
        file.write("\nStarted at:  "+ str(start_time) + "\nFinished at (Date-Time):   " + str(datetime.now().strftime("%D:  %H:%M:%S")))
        file.close()
        return




def training_phase():
    for topK in x_tick_rnd_topK:
        for shrink in x_tick_rnd_shrink:
            
            #x_tick.append("topk {}, shrink {}".format(topK, shrink))
            
            collaborative_recommender_none.fit(shrink=shrink, topK=topK)
            collaborative_recommender_BM25.fit(shrink=shrink, topK=topK,feature_weighting="BM25")
            collaborative_recommender_TF_IDF.fit(shrink=shrink,topK=topK, feature_weighting="TF-IDF")
            
            result_df, _ = evaluator_test.evaluateRecommender(collaborative_recommender_none)
            collaborative_None_MAP.append(result_df.loc[10]["MAP"])
            order_MAP("None",result_df.loc[10]["MAP"],shrink,topK)

            result_df, _ = evaluator_test.evaluateRecommender(collaborative_recommender_BM25)
            collaborative_BM25_MAP.append(result_df.loc[10]["MAP"])
            order_MAP("BM25",result_df.loc[10]["MAP"],shrink,topK)

            result_df, _ = evaluator_test.evaluateRecommender(collaborative_recommender_TF_IDF)
            collaborative_TF_IDF_MAP.append(result_df.loc[10]["MAP"])
            order_MAP("TF-IDF",result_df.loc[10]["MAP"],shrink,topK)

    save_data(phase="Training")  
    return




#Define the validation phase based on the best value acquired in the test phase, stored in the arrays
def validation_phase():
    start_time=datetime.now().strftime("%D:  %H:%M:%S")

    evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    #knnn contenet filter recomennded none feature weighting
    content_recommender_none = ItemKNNCFRecommender(URM_validation)
    
    #knnn contenet filter recomennded BM25 feature weighting
    content_recommender_BM25 = ItemKNNCFRecommender(URM_validation)
    
    #knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = ItemKNNCFRecommender(URM_validation)

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
    start_time=datetime.now().strftime("%D:  %H:%M:%S")
   
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    #knnn contenet filter recomennded none feature weighting
    content_recommender_none = ItemKNNCFRecommender(URM_test)

    #knnn contenet filter recomennded BM25 feature weighting
    content_recommender_BM25 = ItemKNNCFRecommender(URM_test)
    
    #knnn contenet filter recomennded TF_IDF feature weighting
    content_recommender_TF_IDF = ItemKNNCFRecommender(URM_test)
    

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






#Keep the reference to the BestMAP in each phase
Best_MAP=[]
#Keep the referenceto the Model Type(No weigh, BM25, TF-IDF) sorted as the best MAP
Model_type=[]
#Keep the reference to the shrink parameter sorted as the best MAP
Best_Shrink=[]
#Keep the reference to the topK paramter sorted as the bet MAP
Best_topK=[]
#Parameter that declare how many of the best parameter to save, it will be the number of loops for the validantion and test phase
max_length_best=25

#Variable for the num of parameter for shrink and topKin the test phase, the number of loops will be this number squared
size_parameter=25
#Start timeb
start_time=datetime.now().strftime("%D:  %H:%M:%S")



#random search with the log uniform
from scipy.stats import loguniform

x_tick_rnd_topK = loguniform.rvs(10, 500, size=size_parameter).astype(int)
x_tick_rnd_topK.sort()
x_tick_rnd_topK = list(x_tick_rnd_topK)

x_tick_rnd_shrink = loguniform.rvs(10, 500, size=size_parameter).astype(int)
x_tick_rnd_shrink.sort()
x_tick_rnd_shrink = list(x_tick_rnd_topK)


URM_all = load_URM()

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)


evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

#knnn contenet filter recomennded none feature weighting
collaborative_recommender_none = ItemKNNCFRecommender(URM_train)
collaborative_None_MAP = []

#knnn contenet filter recomennded BM25 feature weighting
collaborative_recommender_BM25 = ItemKNNCFRecommender(URM_train)
collaborative_BM25_MAP = []

#knnn contenet filter recomennded TF_IDF feature weighting
collaborative_recommender_TF_IDF = ItemKNNCFRecommender(URM_train)
collaborative_TF_IDF_MAP = []


training_phase()
validation_phase()
testing_phase()


