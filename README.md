# RecSys-Challenge-2022-Polimi

### Resources 

https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi

### Competitions 
* RecSys challenge 2020 https://www.kaggle.com/competitions/recommender-system-2020-challenge-polimi/overview/description

* *RecSys challenge 2021 https://www.kaggle.com/competitions/recommender-system-2021-challenge-polimi/overview/description

* RecSys challenge 2022 https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi


### Colleagues past works

* https://github.com/Alexdruso/recsys-challenge-2020-polimi

* https://github.com/SamueleMeta/recommender-systems

* https://github.com/MathyasGiudici/polimi-recsys-challenge/tree/master/2019

* https://github.com/Menta99/RecSys2021_Mainetti_Menta

## Ideas

- you can stack to ICM also the users as feaures of the items URM + ICM (Works only with  ItemBasedKNN and not the UserBased)
- ItemKNN_CFBF_Hybrid_Recommender cambia ICM_genger in input per poter accettare più ICM e ICM_weight come lista di weight per ogni ICM
- Features for new ICM:
- the one that has seen it most
- the number of users that have seen it)

----------------
# Notes

## Multiple features

    ICM_all= concat(... all ICMs) #(concat the columns)

    sps.vstack([URM_train, ICM_all.T])


## Hybrid models

EX: Usa due modelli che ritornano la stessa struttura (ex Similarity Matrix)\

    itemKNNCF = ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit()
    itemKNNCF.W_sparse
    
    P3alpha = P3alphaRecommender(URM_train)
    P3alpha.fit()
    P3alpha.W_sparse


Poi applica una weighted sum\


    alpha = 0.7
    new_similarity = (1 - alpha) * itemKNNCF.W_sparse + alpha * P3alpha.W_sparse # Weighted Sum


E usa la classe ItemKNNCustomSimilarityRecommender per applicarci la nuova similarity Matrix

    from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
    
    recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
    recommender_object.fit(new_similarity)
    
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
    result_df



#### Compute hybrid with scores
Same as before 

    alpha = 0.7

    new_item_scores = alpha * item_scores_itemknn + (1 - alpha) * item_scores_puresvd
    new_item_scores

#### Compute hybrid with rating prediction + Ranking
USE DifferentLossScoresHybridRecommender


    recommender_object = DifferentLossScoresHybridRecommender(URM_train, funk_svd_recommender, slim_bpr_recommender)

    for norm in [1, 2, np.inf, -np.inf]:

        recommender_object.fit(norm, alpha = 0.66)
    
        result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
        print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))


PROCEDURE:

Ask each recommender to compute the score then define a coefficient and interpolate:


    funk_svd_score = funk_svd_recommender._compute_item_score(user_id)
    slim_bpr_score = slim_bpr_recommender._compute_item_score(user_id).flatten()


Then normalize :

    l1_funk_svd = LA.norm([funk_svd_score], 1)
    l1_funk_svd_scores = funk_svd_score / l1_funk_svd
    
    l2_funk_svd = LA.norm([funk_svd_score], 2)
    l2_funk_svd_scores = funk_svd_score / l2_funk_svd
    
    linf_funk_svd = LA.norm(funk_svd_score, np.inf)
    linf_funk_svd_scores = funk_svd_score / linf_funk_svd
    
    lminusinf_funk_svd = LA.norm(funk_svd_score, -np.inf)
    lminusinf_funk_svd_scores = funk_svd_score / lminusinf_funk_svd
    
    
    l1_slim_bpr = LA.norm(slim_bpr_score, 1)
    l1_slim_bpr_scores = slim_bpr_score / l1_slim_bpr
    
    l2_slim_bpr = LA.norm(slim_bpr_score, 2)
    l2_slim_bpr_scores = slim_bpr_score / l2_slim_bpr
    
    linf_slim_bpr = LA.norm(slim_bpr_score, np.inf)
    linf_slim_bpr_scores = slim_bpr_score / linf_slim_bpr
    
    lminusinf_slim_bpr = LA.norm(slim_bpr_score, -np.inf)
    lminusinf_slim_bpr_scores = slim_bpr_score / lminusinf_slim_bpr

And at the end merge the models:


lambda_weights = 0.66

    l1_new_scores = lambda_weights * l1_slim_bpr_scores + (1 - lambda_weights) * l1_funk_svd_scores
    l2_new_scores = lambda_weights * l2_slim_bpr_scores + (1 - lambda_weights) * l2_funk_svd_scores
    linf_new_scores = lambda_weights * linf_slim_bpr_scores + (1 - lambda_weights) * linf_funk_svd_scores
    lminusinf_new_scores = lambda_weights * lminusinf_slim_bpr_scores + (1 - lambda_weights) * lminusinf_funk_svd_scores



### Tips
* Start with the best model and set a weight and normalization, let's say 1.0 and l1;
* Add the second-best model and try to create a hybrid of the two optimizing only weight and normalization of the model you are trying to add;
* Once the optimization concludes, check if the hybrid is better than before or not. If it is better keep the new hybrid, if not remove the second-best model.
* Continue trying to add the third-best, fourth-best and so on...
* If you are lucky every now and then a new model can be added and the quality improves. (again, result not guaranteed)

## Segmentation

    import matplotlib.pyplot as plt
    %matplotlib inline
    
    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
    from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
    
    MAP_recommender_per_group = {}
    
    collaborative_recommender_class = {"TopPop": TopPop,
    "UserKNNCF": UserKNNCFRecommender,
    "ItemKNNCF": ItemKNNCFRecommender,
    "P3alpha": P3alphaRecommender,
    "RP3beta": RP3betaRecommender,
    "PureSVD": PureSVDRecommender,
    "NMF": NMFRecommender,
    "FunkSVD": MatrixFactorization_FunkSVD_Cython,
    "SLIMBPR": SLIM_BPR_Cython,
    }
    
    content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender,
    "ItemKNNCFCBF": ItemKNN_CFCBF_Hybrid_Recommender
    }
    
    recommender_object_dict = {}
    
    for label, recommender_class in collaborative_recommender_class.items():
    recommender_object = recommender_class(URM_train)
    recommender_object.fit()
    recommender_object_dict[label] = recommender_object
    
    for label, recommender_class in content_recommender_class.items():
    recommender_object = recommender_class(URM_train, ICM_genres)
    recommender_object.fit()
    recommender_object_dict[label] = recommender_object
    cutoff = 10
    
    for group_id in range(0, 20):
    
        start_pos = group_id*block_size
        end_pos = min((group_id+1)*block_size, len(profile_length))
        
        users_in_group = sorted_users[start_pos:end_pos]
        
        users_in_group_p_len = profile_length[users_in_group]
        
        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id, 
            users_in_group.shape[0],
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))
        
        
        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]
        
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)
        
        for label, recommender in recommender_object_dict.items():
            result_df, _ = evaluator_test.evaluateRecommender(recommender)
            if label in MAP_recommender_per_group:
                MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
            else:
                MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]
        
    _ = plt.figure(figsize=(16, 9))
    for label, recommender in recommender_object_dict.items():
    results = MAP_recommender_per_group[label]
    plt.scatter(x=np.arange(0,len(results)), y=results, label=label)
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.legend()
    plt.show()

Guarda i gruppi nel plot che hanno lo stesso comportamento (Pallini colorati posizionati nello stesso modo)