from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Recommender_utils import check_matrix
import numpy as np
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

class RP3_SLIM_BPR(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "SLIM_ITEMKNNCF"

    def __init__(self, URM_train):
        super(RP3_SLIM_BPR, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.SLIM_BPR = SLIM_BPR_Cython(URM_train)
        self.RP3beta = RP3betaRecommender(URM_train )

    def fit(self, topK_CF=1000, shrink_CF=1000, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.5,
            topK=319, learning_rate=0.001  , n_epochs=300,lambda1=0.150,lambda2=0.33, norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.SLIM_BPR.fit()
        self.RP3beta.fit(topK=582,alpha=0.6114597042322002, beta=0.19455372936603404)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        item_scores1 = self.SLIM_BPR._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.RP3beta._compute_item_score(user_id_array, items_to_compute)

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