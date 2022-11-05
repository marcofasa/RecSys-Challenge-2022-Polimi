import os

import pandas as pd
import Utils.Reader as Read

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


#ItemKNNCBFRecommender(URM_train=URM_train,ICM_train=ICM_all)
from Utils.Writer import Writer,NameRecommender

a=Writer(NameRecommender.SLIM_BPR,topK=319,learning_rate=0.001,lambda1=0.01500,lambda2=0.330,URM=URM_train,n_epochs=350)
a.makeSubmission()