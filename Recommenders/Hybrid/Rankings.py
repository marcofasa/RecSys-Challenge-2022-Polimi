from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Recommender_utils import check_matrix
import numpy as np
from numpy import linalg as LA

class Rankings(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is SLIM
    """

    RECOMMENDER_NAME = "SLIM_ITEMKNNCF"

    def __init__(self, URM_train):
        super(Rankings, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.URM_rewatches= URM_train
        self.itemKNNCF = ItemKNNCFRecommender(URM_train)
        self.SLIM = SLIM_BPR_Cython(URM_train)

    def fit(self, topK_CF=343, shrink_CF=488, similarity_CF='cosine', normalize_CF=True,
            feature_weighting_CF="TF-IDF", alpha=0.7,
            topK=319, learning_rate=0.001  , n_epochs=300,lambda1=0.150,lambda2=0.33, norm_scores=True):
        self.alpha = alpha
        self.norm_scores = norm_scores
        self.itemKNNCF.fit(topK=topK_CF, shrink=shrink_CF, similarity=similarity_CF,
                            feature_weighting=feature_weighting_CF)
        self.SLIM.fit(topK=topK,epochs=n_epochs,lambda_i=lambda1,lambda_j=lambda2,learning_rate=learning_rate)

    def recomendation_ranking(self, user_id_array):
        final_raccomandation = {}

        if type(user_id_array)==int:

            final_raccomandation_user = []
            counter_SLIM = 0
            counter_Item = 0
            SLIM_recomandation = self.SLIM.recommend(user_id_array=user_id_array, cutoff=10)
            Item_recomandation = self.itemKNNCF.recommend(user_id_array=user_id_array, cutoff=10)
            for i in range(9):
                if i % 2 == 0:
                    while (final_raccomandation_user.count((Item_recomandation[counter_Item]) )!= 0):
                        counter_Item += 1
                    final_raccomandation_user.append(Item_recomandation[counter_Item])
                    counter_Item += 1
                else:
                    if (final_raccomandation_user.count(SLIM_recomandation[counter_SLIM]) > 0) or counter_SLIM > 3:
                        while (final_raccomandation_user.count((Item_recomandation[counter_Item]) )!= 0):
                                counter_Item += 1
                        final_raccomandation_user.append(Item_recomandation[counter_Item])
                        counter_Item += 1
                    else:


                        final_raccomandation_user.append(SLIM_recomandation[counter_SLIM])
                        counter_SLIM += 1
            return final_raccomandation_user
        else:
            for user_id in user_id_array:
                final_raccomandation_user = []
                counter_SLIM = 0
                counter_Item = 0
                SLIM_recomandation = self.SLIM.recommend(user_id_array=user_id, cutoff=10)
                Item_recomandation = self.itemKNNCF.recommend(user_id_array=user_id, cutoff=10)
                for i in range(9):
                    if i%2==0:
                        while (final_raccomandation_user.count((Item_recomandation[counter_Item])) != 0):
                            counter_Item += 1
                        final_raccomandation_user.append(Item_recomandation[counter_Item])
                        counter_Item += 1
                    else:
                        if (final_raccomandation_user.count((SLIM_recomandation[counter_SLIM]))!= 0 or counter_SLIM > 3):
                            while (final_raccomandation_user.count((Item_recomandation[counter_Item]) )!= 0):
                                counter_Item += 1
                            final_raccomandation_user.append(Item_recomandation[counter_Item])
                            counter_Item += 1
                        else:
                            final_raccomandation_user.append(SLIM_recomandation[counter_SLIM])
                            counter_SLIM += 1
                final_raccomandation[user_id]=final_raccomandation_user

            return final_raccomandation







