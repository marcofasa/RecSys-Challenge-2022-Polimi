import scipy.sparse as sp
import pandas as pd
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Utils.Evaluator import EvaluatorHoldout

#pd=Read.merge(Read.read_ICM_type("csr",matrix_path=matrix_path),Read.read_ICM_length("csr", matrix_path=os.path.join(dirname, "data/data_ICM_length.csv")))

#URM=Read.read_train_csr(matrix_path=matrix_path,stats=True)


import os
from Utils import Reader
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname,  "data/interactions_and_impressions.csv")
ICM_path=os.path.join(dirname,  "data/data_ICM_type.csv")
ICM_path_length=os.path.join(dirname,  "data/data_ICM_length.csv")


rewatches_path=os.path.join(dirname, "data/rewatches_std_normalized.csv")

URM_rewatches=pd.read_csv(rewatches_path, sep=",",
                      skiprows=1,
                      header=None,
                      dtype={0: int, 1: int, 2: float},
                      engine='python')

columns=["UserID","ItemID","data"]
URM_rewatches.columns=columns
print(URM_rewatches)
dictionary={
    0: 1,
    1: 5
}



URM_rewatches = sp.coo_matrix((URM_rewatches[columns[2]].values,
                         (URM_rewatches[columns[0]].values, URM_rewatches[columns[1]].values)
   
                      ))
URM_rewatches.tocsr()



'''
negative_path = os.path.join(dirname, "data/extended.csv")

URM_rewatches = pd.read_csv(rewatches_path, sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: int},
                            engine='python')

columns = ["UserID", "ItemID", "data"]
URM_rewatches.columns = columns
print(URM_rewatches)
URM_rewatches = sp.coo_matrix((URM_rewatches[columns[2]].values,
                               (URM_rewatches[columns[0]].values, URM_rewatches[columns[1]].values)

                               ))
URM_rewatches.tocsr()

dictionary={
    0: 1,
    1: 5
}
'''
#URM=Reader.read_train_csr(matrix_path=matrix_path, switch=False, dictionary=dictionary,column="Data", display=True)


import scipy.sparse as sps




#URM_train, ICM_train=Reader.get_URM_ICM_Type(matrix_path_URM=matrix_path,matrix_path_ICM_type=ICM_path)
URM_train=Reader.read_train_csr(matrix_path=matrix_path)
#URM_rewatches, URM_test = split_train_in_two_percentage_global_sample(URM_rewatches,0.7)



# knnn contenet filter recomennded none feature weighting
from Utils.Writer import NameRecommender
from Utils.Writer import Writer
from Recommenders.Hybrid.Rankings import Rankings

from Recommenders.Hybrid.P3_ITEMKNNCF import P3_ITEMKNNCF
from  Recommenders.Hybrid.FirstLayer import FirstLayer
a=Writer(NameRecommender.USER_ITEM,URM=URM_train)
a.makeSubmission()



'''
ICM_genres = ICM_train

stacked_URM = sps.vstack([URM_train, ICM_genres.T])
stacked_URM = sps.csr_matrix(stacked_URM)

stacked_ICM = sps.csr_matrix(stacked_URM.T)

print(stacked_URM)
print(stacked_ICM)

recommender_ItemKNNCF = ItemKNNCFRecommender(stacked_URM)
recommender_ItemKNNCF.fit()

evaluator = EvaluatorHoldout(URM_test_list=URM_validation, cutoff_list=[10], isRanking=True)

result_df, _ = evaluator.evaluateRecommender(recommender_ItemKNNCF)
print("This is the MAP for only URM stacked:" + str(result_df.loc[10]["MAP"]))

recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_train, stacked_ICM)
recommender_ItemKNNCBF.fit()
'''

import numpy as np
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Recommenders.Hybrid.ITEMKNNCF_SLIM_BPR import ITEMKNNCF_SLIM_BPR
from Recommenders.Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
URM_train, URM_test= split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.90)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

#RECOMMENDER=ItemUserHybridKNNRecommender(URM_train=URM_train)
    #ITEMKNNCF_SLIM_BPR(URM_train=URM_train, URM_rewatches=URM_rewatches)
RECOMMENDER=FirstLayer(URM_train=URM_train,URM_rewatches=URM_rewatches)
evaluator = EvaluatorHoldout(URM_test_list=URM_test, cutoff_list=[10], isRanking=False)
for norm in [ np.inf]:
    RECOMMENDER.fit()
    result_df, _ = evaluator.evaluateRecommender(RECOMMENDER)
    print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))








#result_df, _ = evaluator.evaluateRecommender(RECOMMENDER)
#print("This is the MAP for only URM and ICM stacked: " + str(result_df.loc[10]["MAP"]))


#Hybrid= FirstLayer(URM_train=URM_train,URM_rewatches=URM_rewatches)
#Hybrid.fit( )


