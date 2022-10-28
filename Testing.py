#!/usr/bin/env python3



import numpy as np
import matplotlib.pyplot as pyplot 

from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python


dataReader = Movielens10MReader()
dataset = dataReader.load_data()


URM_all = dataset.get_URM_all()

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)




evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

#declaring the icm matrix
ICM_all = dataset.get_loaded_ICM_dict()["ICM_all"]


#knn contenet based filter item-based
content_recommender = SLIM_BPR_Python(URM_train)
content_MAP = []


#knnn contenet filter recomennded
collaborative_MAP = []

#random search with the log uniform 
from scipy.stats import loguniform

x_tick_rnd_topK = loguniform.rvs(10, 500, size=11).astype(int)
x_tick_rnd_topK.sort()
x_tick_rnd_topK = list(x_tick_rnd_topK)


x_tick_rnd_shrink = loguniform.rvs(10, 500, size=11).astype(int)
x_tick_rnd_shrink.sort()
x_tick_rnd_shrink = list(x_tick_rnd_topK)


x_tick=[]



#random search
for topK in x_tick_rnd_topK:
    for shrink in x_tick_rnd_shrink:
        
        x_tick.append("topk {}, shrink {}".format(topK, shrink))
        
        content_recommender.fit(shrink=shrink, topK=topK)
        collaborative_recommender.fit(shrink=shrink, topK=topK)
        
        result_df, _ = evaluator_test.evaluateRecommender(content_recommender)
        content_MAP.append(result_df.loc[10]["MAP"])
        
        result_df, _ = evaluator_test.evaluateRecommender(collaborative_recommender)
        collaborative_MAP.append(result_df.loc[10]["MAP"])

pyplot.plot(x_tick, collaborative_MAP, label="Collaborative")
pyplot.plot(x_tick, content_MAP, label="Content")   

pyplot.ylabel('Similarity')
pyplot.xlabel('Sorted values')
pyplot.legend()
pyplot.show()


#Lets have a look at how the reccomandation are distribuetded

x_tick = np.arange(URM_all.shape[1])
counter_content = np.zeros(URM_all.shape[1])
counter_collaborative = np.zeros(URM_all.shape[1])

for user_id in range(URM_all.shape[0]):
    recs = collaborative_recommender.recommend(user_id)[:10]
    counter_collaborative[recs] += 1
    
    recs = content_recommender.recommend(user_id)[:10]
    counter_content[recs] += 1
    
    if user_id % 10000 == 0:
        print("Recommended to user {}/{}".format(user_id, URM_all.shape[0]))



sorted_items = np.argsort(-counter_collaborative)
       
pyplot.plot(x_tick, counter_content[sorted_items], label = "Content")
pyplot.plot(x_tick, counter_collaborative[sorted_items], label = "Collaborative")

pyplot.ylabel('Number of recommendations')
pyplot.xlabel('Items')
pyplot.legend()
pyplot.show()