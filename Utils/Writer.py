from enum import Enum
import numpy as np
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import os
import csv
import pandas as pd
class NameRecommender(Enum):
     ItemKNNCFRecommenderNone= "ItemKNNCFRecommenderNone"
     ItemKNNCFRecommenderBM25 = "ItemKNNCFRecommenderBM25"
     ItemKNNCFRecommenderTF_IDF="ItemKNNCFRecommenderTF_IDF"
class Writer(object):

    def __init__(self,NameRecommender,URM,topK,shrink):
        self.NameRecommender=NameRecommender
        self.URM=URM
        self.topK=topK
        self.shrink=shrink

        if(self.NameRecommender.name=="ItemKNNCFRecommenderNone"):
            self.Recommender=ItemKNNCFRecommender(URM=self.URM)
            self.Recommender.fit(topK=self.topK,shrink=self.topK)

        if (self.NameRecommender.name == "ItemKNNCFRecommenderBM25"):
            self.Recommender = ItemKNNCFRecommender(URM=self.URM)
            self.Recommender.fit(topK=self.topK, shrink=self.topK, feature_weighting="BM25")

        if (self.NameRecommender.name == "ItemKNNCFRecommenderTF_IDF"):
            self.Recommender = ItemKNNCFRecommender(URM_train=self.URM)
            self.Recommender.fit(topK=self.topK, shrink=self.topK, feature_weighting="TF-IDF")


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

        with open(write_path,"w+") as d:
            d.write("user_id,item_id\n")
            writer = csv.writer(d)

            for i in range(len(user_array)):

                d.write(str(user_array[i]) + ", ")
                recommandations=self.Recommender.recommend(user_id_array=user_array[i], cutoff=10)
                array=np.asarray(recommandations)
                writer.writerow(array)



