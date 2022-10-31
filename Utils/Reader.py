import math
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sps
from matplotlib import pyplot
from tqdm import tqdm

from Utils.Evaluator import EvaluatorHoldout

columns = ["UserID", "ItemID", "Interaction", "Data"]


def read_train_csr(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                   stats=False, preprocess=0, display=False, saving=False, progress=True):
    n_items = 0
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: str, 3: int},
                            engine='python')
    matrix_df.columns = columns

    # flipping 0 and 1 (from now on:
    # 0--> just description see interaction
    # 1--> real interaction
    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0})
    print(len(matrix_df))

    if preprocess > 0:
        print("Preprocessing..")
        df_preprocess(matrix_df, saving=True, mode=preprocess)

    if display:
        print("Displaying..")
        print(matrix_df.head())

    # stats
    if stats:
        n_items = df_stats(matrix_df, progress)

    matrix = sps.coo_matrix((matrix_df[columns[3]].values,
                             (matrix_df[columns[0]].values, matrix_df[columns[1]].values)
                             ))

    # TODO also consider other shows appeared
    # if is 1 (just description read) is less important

    if matrix_format == "csr":
        if not stats:
            return matrix.tocsr()
        else:
            csr_stats(matrix.tocsr(), n_items)
    else:
        return matrix.tocsc()


def df_preprocess(df, saving=True, mode=0, progress=True):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # print(index)
        print(len(df))

        displayList = []
        if type(row[columns[2]]) == str:
            displayList = row[columns[2]].split(",")
            displayList = [eval(i) for i in displayList]

        # counting other TV series displayed
        if mode == 1:
            df.loc[index, 'Displayed'] = len(displayList)  # insert to the new column
        elif mode == 2:
            userid = row[columns[0]]
            for item in displayList:
                df.loc[len(df.index)] = [userid, item, None, -1]
        # print("OKOK")

    if saving:
        save(df, "out_" + str(mode))


def df_stats(dataframe):
    userID_unique = dataframe["UserID"].unique()
    itemID_unique = dataframe["ItemID"].unique()

    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    n_interactions = len(dataframe)
    print("------ITEM POPULARITY------")

    print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemID_unique), max(userID_unique)))
    print("Average interactions per user {:.2f}".format(n_interactions / n_users))
    print("Average interactions per item {:.2f}\n".format(n_interactions / n_items))

    print("Sparsity {:.2f} %".format((1 - float(n_interactions) / (n_items * n_users)) * 100))
    print("--------------------")

    return n_items


def csr_stats(csr, n_items):
    item_popularity = np.ediff1d(csr.tocsc().indptr)
    item_popularity = np.sort(item_popularity)
    print("------ITEM POPULARITY------")
    print(item_popularity)
    pyplot.plot(item_popularity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Sorted Item')
    pyplot.show()

    ten_percent = int(n_items / 10)
    print("------AVG PER-ITEM------")

    print("Average per-item interactions over the whole dataset {:.2f}".
          format(item_popularity.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
          format(item_popularity[-ten_percent:].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
          format(item_popularity[:ten_percent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
          format(item_popularity[int(n_items * 0.45):int(n_items * 0.55)].mean()))

    print("Number of items with zero interactions {}".format(np.sum(item_popularity == 0)))
    print("---------------------")

    user_activity = np.ediff1d(csr.tocsr().indptr)
    user_activity = np.sort(user_activity)

    pyplot.plot(user_activity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Sorted User')
    pyplot.show()


def read_ICM_length(matrix_format="csr", clean=True,matrix_path="../data/data_ICM_type.csv"):
    df = pd.read_csv(filepath_or_buffer=matrix_path,
                     sep=",",
                     skiprows=1,
                     header=None,
                     dtype={0: int, 1: int, 2: int},
                     engine='python')
    df.columns = ['ItemID', 'feature', 'Episodes']

    # Since there's only one feature the FeatureID column is useless (all zeros)
    if clean:
        df = df.drop('feature', axis=1)
    df.set_index('ItemID', inplace=True)
    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=df, columns=["EPLength"]).to_numpy())
    else:
        return sps.csc_matrix(pd.DataFrame(data=df, columns=["EPLength"]).to_numpy())


def read_ICM_type(matrix_format="csr", clean=True,matrix_path="../data/data_ICM_type.csv"):
    df = pd.read_csv(filepath_or_buffer=matrix_path,
                     sep=",",
                     skiprows=1,
                     header=None,
                     dtype={0: int, 1: int, 2: int},
                     engine='python')
    df.columns = ['ItemID', 'TypeID', 'data']

    # Since there's only one feature the data column is useless (all 1s)
    if clean:
        df = df.drop('data', axis=1)
    df.set_index('ItemID', inplace=True)
    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())
    else:
        return sps.csc_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())


def get_user_segmentation(URM_train, URM_val, start_pos, end_pos):
    profile_length = np.ediff1d(URM_train.indptr)
    sorted_users = np.argsort(profile_length)
    users_in_group = sorted_users[start_pos:end_pos]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    return EvaluatorHoldout(URM_val, cutoff_list=[10], ignore_users=users_not_in_group)


def merge(ICM_a, ICM_b):
    return sps.hstack([ICM_a, ICM_b])


def save(data, name, path="../output/"):
    data.to_csv(path + name + '.csv')



################

if __name__ == '__main__':
    read_train_csr(preprocess=2, saving=True)
