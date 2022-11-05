import os
import scipy.sparse as sp
import pandas as pd
import Utils.Reader as Read


from FirstIbrid import FirstIbrid
from Utils.Reader import EvaluatorHoldout


#pd=Read.merge(Read.read_ICM_type("csr",matrix_path=matrix_path),Read.read_ICM_length("csr", matrix_path=os.path.join(dirname, "data/data_ICM_length.csv")))
from Recommenders.MatrixFactorization import SVDFeatureRecommender
from Recommenders.MatrixFactorization import PureSVDRecommender

#URM=Read.read_train_csr(matrix_path=matrix_path,stats=True)


from  Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import os
from Utils import Reader
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname,  "data/interactions_and_impressions.csv")
ICM_path=os.path.join(dirname,  "data/data_ICM_type.csv")
ICM_path_length=os.path.join(dirname,  "data/data_ICM_length.csv")
URM_train=Reader.read_train_csr(matrix_path=matrix_path)


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


#ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICM_all)
from Utils.Writer import Writer,NameRecommender

a=Writer(NameRecommender.Hybrid,topK=319,learning_rate=0.001,lambda1=0.01500,lambda2=0.330,URM=URM_train,n_epochs=350,URM_rewatches=URM_rewatches)
a.makeSubmission()
'''
recomender = FirstIbrid(URM_train=URM_train, URM_rewatches=URM_rewatches)
recomender.fit()
evaluator = EvaluatorHoldout(URM_test_list=URM_train, cutoff_list=[10])

result_df, _ = evaluator.evaluateRecommender(recomender)
print("This is the MAP:" + str(result_df.loc[10]["MAP"]))
'''
