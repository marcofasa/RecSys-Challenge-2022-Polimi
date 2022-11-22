import scipy.sparse as sp
import pandas as pd
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_double

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
matrix_extended=os.path.join(dirname,  "data/extended.csv")
ICM_path=os.path.join(dirname,  "data/data_ICM_type.csv")
ICM_path_length=os.path.join(dirname,  "data/data_ICM_length.csv")

#normalize map: 1444
#reqatche only normazlie: 1.43

#rewatches_path=os.path.join(dirname, "data/rewatches.csv")
'''
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

URM_train=Reader.read_train_csr(matrix_path=matrix_path)

'''
import scipy.sparse as sps

#URM_train, ICM_train= Reader.get_URM_ICM_Type(matrix_path_URM=matrix_path, matrix_path_ICM_type=ICM_path)
stacked_path = os.path.join(dirname, "data/Type_1Hot.csv")


''''
URM_train= pd.read_csv(matrix_path, sep=",",
                        skiprows=1,
                       header=None,
                       engine="python")

ICM_length_OneHot = pd.read_csv(stacked_path, sep=",",
                            skiprows=1,
                            header=None,
                            engine='python')

all_item_indices = pd.concat([URM_train[1], ICM_length_OneHot[0]], ignore_index=True)
mapped_id, original_id = pd.factorize(all_item_indices.unique())

print("Unique ItemID in the URM and ICM are {}".format(len(original_id)))

item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

#URM_train[1] = URM_train[1].map(item_original_ID_to_index)

ICM_length_OneHot[0] = ICM_length_OneHot[0].map(item_original_ID_to_index)


ICM_length_OneHot=sps.csr_matrix(ICM_length_OneHot)
ICM_length_OneHot.tocsr()
print(ICM_length_OneHot)
'''

#ICM_length_OneHot = sp.coo_matrix((ICM_length_OneHot.values))
#ICM_length_OneHot.tocsr()





#stacked_ICM = sps.csr_matrix(stacked_URM.T)

#print(stacked_URM)
#print(stacked_ICM)



import scipy.sparse as sps




#URM_train, ICM_train=Reader.get_URM_ICM_Type(matrix_path_URM=matrix_path,matrix_path_ICM_type=ICM_path)
#URM_train, ICM_train=Reader.get_URM_ICM_Type(matrix_path_URM=matrix_path, matrix_path_ICM_type=ICM_path)
#URM_rewatches, URM_test = split_train_in_two_percentage_global_sample(URM_rewatches,0.7)



# knnn contenet filter recomennded none feature weighting
from Utils.Writer import NameRecommender
from Utils.Writer import Writer
from Recommenders.Hybrid.Rankings import Rankings
#URM_train=Reader.read_train_csr(matrix_path=matrix_path)

from Recommenders.Hybrid.P3_RP3 import P3_RP3



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
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from Recommenders.Hybrid.P3_RP3 import P3_RP3
from Recommenders.Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
#URM_rewatches,ICM_train=Reader.get_URM_ICM_Type_Extended(matrix_path_URM=matrix_extended,matrix_path_ICM_type=ICM_path)

#URM_rewatches, URM_test= split_train_in_two_percentage_global_sample(URM_rewatches, train_percentage=0.70)

URM_train=Reader.read_train_csr(matrix_path=matrix_path,stats=True, value=True)
URM_train_extended=Reader.read_train_csr_extended(matrix_path=matrix_extended)
#URM_train1, URM_test1, URM_train2, URM_test2 =split_train_in_two_percentage_global_sample_double(URM_train_extended,URM_train)
#RM_train,URM_test=split_train_in_two_percentage_global_sample(URM_train_extended,train_percentage=0.7)

a=Writer(NameRecommender.USER_ITEM,URM=URM_train,URM_rewatches=URM_train_extended)
a.makeSubmission()

#URM_train_normal, URM_test = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.70)
#URM_train, URM_validation,URM_train_extended, URM_validation_extended=split_train_in_two_percentage_global_sample_double(URM_train,URM_train_extended)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.70)
from  Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.Hybrid.FirstLayer import FirstLayer
from Recommenders.Hybrid.NewHybrid import NewHybrid
evaluator_validation = EvaluatorHoldout(URM_test_list=URM_validation, cutoff_list=[10], isRanking=False)
#evaluator_test= EvaluatorHoldout(URM_test_list=URM_test, cutoff_list=[10], isRanking=False)
from Recommenders.Hybrid.P3_RP3 import P3_RP3


RECOMMENDER = FirstLayer(URM_train=URM_train, URM_extended=URM_train_extended)
RECOMMENDER.fit()
result_df_validation, _ = evaluator_validation.evaluateRecommender(RECOMMENDER)
print(" This is the MAP for validation: {}".format( result_df_validation.loc[10]["MAP"]))


from Recommenders.Hybrid.RP3_ITEMHYBRID import RP3_SLIM_BPR
RECOMMENDER=MatrixFactorization_BPR_Cython(URM_train=URM_train_normal)
RECOMMENDER.fit()
result_df_test, _ = evaluator_test.evaluateRecommender(RECOMMENDER)
print(" This is the MAP for validation: {}".format( result_df_validation.loc[10]["MAP"]))
print(" This is the MAP for testing: {}".format( result_df_test.loc[10]["MAP"]))








#result_df, _ = evaluator.evaluateRecommender(RECOMMENDER)
#print("This is the MAP for only URM and ICM stacked: " + str(result_df.loc[10]["MAP"]))


#Hybrid= FirstLayer(URM_train=URM_train,URM_rewatches=URM_rewatches)
#Hybrid.fit( )


