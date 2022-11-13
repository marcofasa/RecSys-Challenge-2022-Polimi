from Utils.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Utils import Reader
import os
dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname,  "data/interactions_and_impressions.csv")
URM= Reader.read_train_csr(matrix_path)

eval=EvaluatorHoldout(URM,cutoff_list=[10])
rec=SLIM_BPR_Cython(URM_train=URM)
rec.fit(learning_rate=0.001,lambda_j=0.3,lambda_i=0.2,topK=118)
resutl , _ = eval.evaluateRecommender(recommender_object=rec)

print(resutl.loc[10]["MAP"])