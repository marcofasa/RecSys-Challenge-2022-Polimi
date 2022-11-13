from enum import Enum
import numpy as np

from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Hybrid.ITEMKNNCF_SLIM_BPR import ITEMKNNCF_SLIM_BPR
from Recommenders.Hybrid.FirstLayer import FirstLayer
from Recommenders.Hybrid.Rankings import Rankings
from Recommenders.Hybrid.RP3_ITEMHYBRID import RP3_SLIM_BPR
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Recommenders.Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
import os
import csv
import pandas as pd

class NameRecommender(Enum):
     ItemKNNCFRecommenderNone= "ItemKNNCFRecommenderNone"
     ItemKNNCFRecommenderBM25 = "ItemKNNCFRecommenderBM25"
     ItemKNNCFRecommenderTF_IDF="ItemKNNCFRecommenderTF_IDF"
     SLIM_BPR="SLIM_BPR"
     Hybrid="Hybrid"
     FirstLayer="FirstLayer"
     Rankings="Rankings"
     P3_ITEMKNNCF="P3_ITEMKNNCF"
     HybridNorm = "HybridNorm"
     USER_ITEM="USER_ITEM"
     ItemKNNCBF="ItemKNNCBF"
class Writer(object):

    def __init__(self,NameRecommender,URM,topK=None,shrink=None,learning_rate=None, lambda1=None,lambda2=None, n_epochs=None,
                 URM_rewatches=None,ICM=None, shrink_CF=None, topk_CF=None, alpha=None, stackedICM=None):
        self.NameRecommender=NameRecommender
        self.URM=URM
        self.topK=topK
        self.shrink=shrink
        self.ICM=ICM
        if(self.NameRecommender.name=="ItemKNNCFRecommenderNone"):
            self.Recommender=ItemKNNCFRecommender(URM=self.URM)
            self.Recommender.fit(topK=self.topK,shrink=self.topK)

        if (self.NameRecommender.name == "ItemKNNCFRecommenderBM25"):
            self.Recommender = ItemKNNCFRecommender(URM=self.URM)
            self.Recommender.fit(topK=self.topK, shrink=self.topK, feature_weighting="BM25")

        if (self.NameRecommender.name == "ItemKNNCFRecommenderTF_IDF"):
            self.Recommender = ItemKNNCFRecommender(URM_train=self.URM)
            self.Recommender.fit(topK=self.topK, shrink=self.topK, feature_weighting="TF-IDF")

        if (self.NameRecommender.name == "SLIM_BPR"):
            self.Recommender = SLIM_BPR_Cython(URM_train=self.URM)
            self.Recommender.fit(topK=self.topK,epochs=n_epochs,lambda_i=lambda1,lambda_j=lambda2,learning_rate=learning_rate)
        if (self.NameRecommender.name == "Hybrid"):
            self.Recommender = ITEMKNNCF_SLIM_BPR(URM_train=self.URM, URM_rewatches=URM_rewatches)
            self.Recommender.fit(topK=self.topK,n_epochs=n_epochs,lambda1=lambda1,lambda2=lambda2,learning_rate=learning_rate,topK_CF=343,shrink_CF=488)
        if (self.NameRecommender.name == "Rankings"):
            self.Recommender = Rankings(URM_train=self.URM)
            self.Recommender.fit()
        if (self.NameRecommender.name == "FirstLayer"):
            self.Recommender = FirstLayer(URM_train=self.URM)
            self.Recommender.fit()
        if(self.NameRecommender.name=="P3_ITEMKNNCF"):
            self.Recommender= RP3_SLIM_BPR(URM_train=URM,URM_rewatches= URM_rewatches)
            self.Recommender.fit()
        if (self.NameRecommender.name == "HybridNorm"):
            self.Recommender = DifferentLossScoresHybridRecommender(URM_train=URM, URM_rewatches=URM_rewatches)
            self.Recommender.fit(norm=np.inf)
        if (self.NameRecommender.name == "USER_ITEM"):
            self.Recommender = ItemUserHybridKNNRecommender(URM_train=URM)
            self.Recommender.fit()
        if (self.NameRecommender.name == "ItemKNNCBF"):
            self.Recommender = ItemKNNCBFRecommender(URM_train=URM, ICM_train=stackedICM)
            self.Recommender.fit()

    def makeSubmission(self):
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(current_dir + "/../")
        read_path=os.path.join(parent_dir,"data/data_target_users_test.csv")

        with open(read_path,"r") as f:
            dataFrame=pd.read_csv(f,skiprows=0)
            dataFrame.columns=["UserID"]
            user_array=np.asarray(dataFrame["UserID"])

        write_path = os.path.join(parent_dir, "Testing_Results/submission.csv")
        print(user_array)
        user_array=user_array.tolist()
        with open(write_path,"w+") as d:
            d.write("user_id,item_list\n")
            writer = csv.writer(d, delimiter=' ')

            for i in range(len(user_array)):

                d.write(str(user_array[i]) + ", ")
   #             if(self.NameRecommender.name!="Rankings"):
                recommandations=self.Recommender.recommend(user_id_array=user_array[i], cutoff=10)
 #               else:
#                    recommandations=self.Recommender.recomendation_ranking(user_id_array=user_array[i])
                array=np.asarray(recommandations)
                writer.writerow(array)



