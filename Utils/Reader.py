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
                   stats=False, preprocess=0, display=False, saving=False):
    n_items = 0
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: str, 3: int},
                            engine='python')
    matrix_df.columns = columns
    mapped_id, original_id = pd.factorize(matrix_df["UserID"].unique())

    print("Unique UserID in the URM are {}".format(len(original_id)))

    all_item_indices = pd.concat([matrix_df["UserID"], matrix_df["UserID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique UserID in the URM and ICM are {}".format(len(original_id)))

    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    # matrix_df.loc[~(matrix_df == 0).all(axis=2)]
    # matrix_df = matrix_df.drop(matrix_df[matrix_df[columns[3]] ==0].index)
    # flipping 0 and 1 (from now on:
    # 0--> just description see interaction
    # 1--> real interaction

    #df_col_normalize(matrix_df,columns[3],{0: 1, 1: 0})
    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0})

    if preprocess > 0:
        print("Preprocessing..")
        df_preprocess(matrix_df, saving=True, mode=preprocess)

    print(len(matrix_df))

    if display:
        print("Displaying..")
        print(matrix_df.head())

    # stats
    if stats:
        n_items = df_stats(matrix_df)

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
            return matrix.tocsr()
    else:
        return matrix.tocsc()


    # colsToChange: The column where replace values
    # valsToPlace: dictionary of vals of kind:
    #  {oldval1: newval1 , oldval2: newval1,...}
def df_col_normalize(df, colToChange, valsToPlace):
    df[colToChange] = df[colToChange].replace(valsToPlace)


def df_preprocess(df, saving=True, mode=0):
    list_to_convert = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # print(index)
        #print(len(df))

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
                list_to_convert.append([userid, item, None, -1])

    if mode == 2:
        df1 = pd.DataFrame(list_to_convert, columns=columns)
        df = df.append(df1)

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


def read_ICM_length(matrix_format="csr", clean=True, matrix_path="../data/data_ICM_type.csv"):
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


def read_ICM_type(matrix_format="csr", clean=True, matrix_path="../data/data_ICM_type.csv"):
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


def save(data, name,fullPath, relativePath="../output/"):
    data.to_csv(relativePath + name + '.csv', index=False)


def get_URM_ICM_Type(matrix_path_URM, matrix_path_ICM_type, matrix_path_ICM_length):
    columns = ["UserID", "ItemID", "Interaction", "Data"]
    n_items = 0
    URM = pd.read_csv(filepath_or_buffer=matrix_path_URM,
                      sep=",",
                      skiprows=1,
                      header=None,
                      dtype={0: int, 1: int, 2: str, 3: int},
                      engine='python')
    URM.columns = columns

    ICM_type = pd.read_csv(filepath_or_buffer=matrix_path_ICM_type,
                           sep=",",
                           skiprows=1,
                           header=None,
                           dtype={0: int, 1: int, 2: int},
                           engine='python')
    ICM_type.columns = ['ItemID', 'FeatureID', 'data']

    ICM_length = pd.read_csv(filepath_or_buffer=matrix_path_ICM_length,
                             sep=",",
                             skiprows=1,
                             header=None,
                             dtype={0: int, 1: int, 2: int},
                             engine='python')
    ICM_length.columns = ['ItemID', 'Episodes', 'Data']

    mapped_id, original_id = pd.factorize(URM["UserID"].unique())

    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    all_item_indices = pd.concat([URM["ItemID"], ICM_type["ItemID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique ItemID in the URM and ICM_length are {}".format(len(original_id)))

    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(ICM_type["FeatureID"].unique())
    feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    print("Unique FeatureID in the URM are {}".format(len(feature_original_ID_to_index)))

    URM["UserID"] = URM["UserID"].map(user_original_ID_to_index)
    URM["ItemID"] = URM["ItemID"].map(item_original_ID_to_index)
    URM[columns[3]] = URM[columns[3]].replace({0: 1, 1: 0})
    # matrix_df.loc[~(matrix_df == 0).all(axis=2)]

    ICM_type["ItemID"] = ICM_type["ItemID"].map(item_original_ID_to_index)
    ICM_type["FeatureID"] = ICM_type["FeatureID"].map(feature_original_ID_to_index)

    n_users = len(user_original_ID_to_index)
    n_items = len(item_original_ID_to_index)
    n_features = len(feature_original_ID_to_index)

    URM_all = sps.csr_matrix((URM[columns[3]].values,
                              (URM[columns[0]].values, URM[columns[1]].values)),
                             shape=(n_users, n_items))

    ICM_all_type = sps.csr_matrix((np.ones(len(ICM_type["ItemID"].values)),
                                   (ICM_type["ItemID"].values, ICM_type["FeatureID"].values)),
                                  shape=(n_items, n_features))

    ICM_all_length = sps.csr_matrix((np.ones(len(ICM_length["Data"].values)),
                                     (ICM_length["ItemID"].values, ICM_length["Episodes"].values)),
                                    shape=(n_items, n_features))

    ICM_all_type.data = np.ones_like(ICM_all_type.data)

    return URM_all, ICM_all_type


################

if __name__ == '__main__':
    read_train_csr(preprocess=2, saving=True)
