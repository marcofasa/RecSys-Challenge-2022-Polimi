import numpy as np
import scipy.sparse as sp
import pandas as pd
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

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


rewatches_path=os.path.join(dirname, "data/rewatches.csv")

URM_rewatches=pd.read_csv(rewatches_path, sep=",",
                      skiprows=1,
                      header=None,
                      dtype={0: int, 1: int, 2: int},
                      engine='python')

columns=["UserID","ItemID","data"]
URM_rewatches.columns=columns
print(URM_rewatches)
URM_rewatches = sp.coo_matrix((URM_rewatches[columns[2]].values,
                         (URM_rewatches[columns[0]].values, URM_rewatches[columns[1]].values)
   
                      ))
URM_rewatches.tocsr()



dictionary={
    0: 1,
    1: 5
}

#URM=Reader.read_train_csr(matrix_path=matrix_path, switch=False, dictionary=dictionary,column="Data", display=True)







URM_train, ICM_train=Reader.get_URM_ICM_Type(matrix_path_URM=matrix_path,matrix_path_ICM_type=ICM_path)
#URM_train=Reader.read_train_csr(matrix_path=matrix_path)
#URM_rewatches, URM_test = split_train_in_two_percentage_global_sample(URM_rewatches,0.7)



# knnn contenet filter recomennded none feature weighting
from Utils.Writer import NameRecommender
from Utils.Writer import Writer
from Recommenders.Hybrid.Rankings import Rankings

from Recommenders.Hybrid.P3_ITEMKNNCF import P3_ITEMKNNCF
from Recommenders.Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender

#a=Writer(NameRecommender.SLIM_BPR,topK=319 , learning_rate=0.001  , n_epochs=300 ,lambda1=0.01578,lambda2=0.32905,URM=URM_rewatches)
#a.makeSubmission()


URM_train, URM_test= split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.60)

'''
Hybrid= FirstLayer(URM_train=URM_train,URM_rewatches=URM_rewatches)
Hybrid.fit( )
evaluator = EvaluatorHoldout(URM_test_list=URM_validation, cutoff_list=[10], isRanking=True)

result_df, _ = evaluator.evaluateRecommender(Hybrid)
print("This is the MAP:" + str(result_df.loc[10]["MAP"]))
'''

Hybrid= DifferentLossScoresHybridRecommender(URM_train=URM_train,URM_rewatches=URM_rewatches)
evaluator_validation=EvaluatorHoldout(URM_validation,cutoff_list=[10])
for norm in [1, 2, np.inf, -np.inf]:

    Hybrid.fit(norm, alpha = 0.66)
    result_df, _ = evaluator_validation.evaluateRecommender(Hybrid)
    print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))
