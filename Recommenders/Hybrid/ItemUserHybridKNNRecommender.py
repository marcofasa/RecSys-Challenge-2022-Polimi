from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Recommender_utils import check_matrix
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
import numpy as np
from numpy import linalg as LA


class ItemUserHybridKNNRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "ITEMCFCBF_ITEMKNNCF"

    def __init__(self, URM_train):
        super(ItemUserHybridKNNRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = ItemKNNCFRecommender(URM_train)
        self.UserKNN = UserKNNCFRecommender(URM_train)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.6,
            topK=402, shrink=644, feature_weighting="TF-IDF",  norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit(topK=topK_CF, shrink=shrink_CF, similarity=similarity_CF,
                            feature_weighting=feature_weighting)
        self.UserKNN.fit(topK=topK,shrink=shrink,feature_weighting=feature_weighting)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_scores1 = self.itemKNNCF._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.UserKNN._compute_item_score(user_id_array, items_to_compute)

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