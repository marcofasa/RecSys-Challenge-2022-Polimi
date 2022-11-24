import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from Recommenders.Recommender_utils import check_matrix
from numpy import linalg as LA

class NewHybrid1(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "SLIM_ITEMKNNCF"

    def __init__(self, URM_train):
        super(NewHybrid1, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = UserKNNCFRecommender(URM_train)
        self.SLIM = RP3betaRecommender(URM_train)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.5,
            topK=319 , learning_rate=0.001  , n_epochs=300 ,lambda1=0.01578,lambda2=0.32905, norm_scores=False):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit()
        self.SLIM.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        item_scores1 = self.itemKNNCF._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.SLIM._compute_item_score(user_id_array, items_to_compute)

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