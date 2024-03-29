from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Recommender_utils import check_matrix
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
import numpy as np

class ITEMKNNCF_ITEMCFCBF(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "ITEMCFCBF_ITEMKNNCF"

    def __init__(self, URM_train, ICM_train):
        super(ITEMKNNCF_ITEMCFCBF, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = ItemKNNCFRecommender(URM_train)
        self.ItemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train,ICM_train)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.7,
            topK=319, shrink=300, feature_weighting="TF-IDF",  norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit(topK=topK_CF, shrink=shrink_CF, similarity=similarity_CF,
                            feature_weighting=feature_weighting)
        self.ItemKNNCFCBF.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        item_scores1 = self.itemKNNCF._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.ItemKNNCFCBF._compute_item_score(user_id_array, items_to_compute)

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