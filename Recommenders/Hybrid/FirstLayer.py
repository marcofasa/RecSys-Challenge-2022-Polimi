from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Hybrid.GraphBasedHybrid import GraphBasedHybrid
from Recommenders.Hybrid.P3_RP3 import P3_RP3
from Recommenders.Hybrid.ItemUserHybridKNNRecommender import ItemUserHybridKNNRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.Hybrid.NewHybrid import NewHybrid
from Recommenders.Hybrid.NewHybrid1 import NewHybrid1
import numpy as np
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Hybrid.RP3_ITEMHYBRID import RP3_SLIM_BPR
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from numpy import linalg as LA
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
class FirstLayer(BaseItemSimilarityMatrixRecommender,IALSRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "ITEMCFCBF_SLIMBPR"

    def  __init__(self, URM_train, URM_extended):
        super(FirstLayer, self).__init__(URM_train)


        self.firstHybrid=ItemUserHybridKNNRecommender(URM_train=URM_train)
        self.secondHybrid=MatrixFactorization_BPR_Cython(URM_train=URM_extended)



    def fit(self,topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=False,
            feature_weighting_CF="TF-IDF", alpha_hybrid=0.3,
            epochs=319 , num_factors=0.001  , alpha=300 ,epsilon=0.01578,confidence_scaling="",reg=0, norm_scores=True,**earlystopping_kwargs):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.norm=2
        self.firstHybrid.fit()
        self.secondHybrid.fit()

        '''
        self.secondHybrid.fit( epochs = epochs,
            num_factors = num_factors,
            confidence_scaling = confidence_scaling,
            alpha = alpha,
            epsilon =  epsilon,
            reg = reg,
            **earlystopping_kwargs)
        '''

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_scores1 = self.firstHybrid._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.secondHybrid._compute_item_score(user_id_array, items_to_compute)

        if self.norm_scores:
            mean1 = np.mean(item_scores1)
            mean2 = np.mean(item_scores2)
            std1 = np.std(item_scores1)
            std2 = np.std(item_scores2)
            if std1 != 0 and std2 != 0:
                item_scores1 = (item_scores1 - mean1) / std1
                item_scores2 = (item_scores2 - mean2) / std2
            '''max1 = item_scores1.max()
            max2 = item_scores2.max()
            item_scores1 = item_scores1 / max1
            item_scores2 = item_scores2 / max2'''
        #print(item_scores1)
        #print(item_scores2)

        item_scores = item_scores1 * self.alpha + item_scores2 * (1 - self.alpha)

        return item_scores

