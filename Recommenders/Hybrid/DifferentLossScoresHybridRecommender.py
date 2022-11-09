from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, URM_rewatches):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = ItemKNNCFRecommender(URM_train)
        self.SLIM = SLIM_BPR_Cython(URM_rewatches)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.7,
            topK=319 , learning_rate=0.001  , n_epochs=300 ,lambda1=0.01578,lambda2=0.32905, norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit(topK=topK_CF, shrink=shrink_CF, similarity=similarity_CF,
                           feature_weighting=feature_weighting_CF)
        self.SLIM.fit(topK=topK,epochs=n_epochs,lambda_i=lambda1,lambda_j=lambda2,learning_rate=learning_rate)


    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights_1 = self.itemKNNCF._compute_item_score(user_id_array)
        item_weights_2 = self.SLIM._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)


        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights
