from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
import numpy as np
from numpy import linalg as LA


class GraphBasedHybrid(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "ITEMCFCBF_ITEMKNNCF"

    def __init__(self, URM_train):
        super(GraphBasedHybrid, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.RP3 = RP3betaRecommender(URM_train)
        self.P3A = P3alphaRecommender(URM_train)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.8,
            topK=402, shrink=644, feature_weighting="TF-IDF",  norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.RP3.fit()
        self.P3A.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_scores1 = self.RP3._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.P3A._compute_item_score(user_id_array, items_to_compute)

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
        print(item_scores1)
        print(item_scores2)

        item_scores = item_scores1 * self.alpha + item_scores2 * (1 - self.alpha)

        return item_scores