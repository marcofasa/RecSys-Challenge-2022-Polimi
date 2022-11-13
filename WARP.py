import os
import time

import numpy as np



import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Utils import Reader

dirname = os.path.dirname(__file__)
matrix_path = os.path.join(dirname, "data/interactions_and_impressions.csv")

URM=Reader.read_train_csr(matrix_path=matrix_path)
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.70)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.70)

alpha = 1e-05
epochs = 70
num_components = 32

warp_model = LightFM(no_components=num_components,
                     loss='warp',
                     learning_schedule='adagrad',
                     max_sampled=100,
                     user_alpha=alpha,
                     item_alpha=alpha)

bpr_model = LightFM(no_components=num_components,
                    loss='bpr',
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)

warp_duration = []
bpr_duration = []
warp_auc = []
bpr_auc = []

for epoch in tqdm(range(epochs)):
    start = time.time()
    warp_model.fit_partial(URM_train, epochs=1)
    warp_duration.append(time.time() - start)
    warp_auc.append(auc_score(warp_model, URM_test, train_interactions=URM_train).mean())

for epoch in tqdm(range(epochs)):
    start = time.time()
    bpr_model.fit_partial(URM_train, epochs=1)
    bpr_duration.append(time.time() - start)
    bpr_auc.append(auc_score(bpr_model, URM_test, train_interactions=URM_train).mean())

x = np.arange(epochs)
plt.plot(x, np.array(warp_auc))
plt.plot(x, np.array(bpr_auc))
plt.legend(['WARP AUC', 'BPR AUC'], loc='upper right')
plt.show()
warp_model = LightFM(no_components=num_components,
                     max_sampled=3,
                    loss='warp',
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)

warp_duration = []
warp_auc = []

for epoch in range(epochs):
    start = time.time()
    warp_model.fit_partial(train, epochs=1)
    warp_duration.append(time.time() - start)
    warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

x = np.arange(epochs)
plt.plot(x, np.array(warp_duration))
plt.legend(['WARP duration'], loc='upper right')
plt.title('Duration')
plt.show()

x = np.arange(epochs)
plt.plot(x, np.array)
