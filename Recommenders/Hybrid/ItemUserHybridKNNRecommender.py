from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Recommender_utils import check_matrix
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
import numpy as np
from numpy import linalg as LA


class ItemUserHybridKNNRecommender(ItemKNNCFRecommender, UserKNNCFRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "ItemUserHybridKNNRecommender"

    def __init__(self, URM_train):
        super(ItemUserHybridKNNRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = ItemKNNCFRecommender(URM_train)
        self.UserKNN = UserKNNCFRecommender(URM_train)


    def fit(self, topK_CF=2000, shrink_CF=1000, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.7,
            topK=1899, shrink=10, feature_weighting="TF-IDF",
            similarity='cosine',normalize=True,  norm_scores=True, **fit_args):

        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit(topK=topK_CF,shrink=shrink_CF,similarity=similarity_CF, normalize=normalize_CF)
        self.UserKNN.fit(topK=topK,shrink=shrink,similarity=similarity, normalize=normalize)
        #super(ItemUserHybridKNNRecommender, self).fit(**fit_args)



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