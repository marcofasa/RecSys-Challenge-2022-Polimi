import math
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sps
from matplotlib import pyplot
from tqdm import tqdm
import os

from Utils.Evaluator import EvaluatorHoldout

columns = ["UserID", "ItemID", "Interaction", "Data"]


def df_col_normalize(df, colToChange, valsToPlace=None):
    if valsToPlace == None:
        # return (df[colToChange]-df[colToChange].mean())/df[colToChange].std() #std normalization
        return (df[colToChange] - df[colToChange].min()) / (df[colToChange].max() - df[colToChange].min())
    else:
        return df[colToChange].replace(valsToPlace)


def df_splitter(columnToDivide, divisionList, dtype={0: int, 1: int, 2: int}, dfPath='../data/rewatches.csv',
                colstoDefine=None, name=""):
    if colstoDefine is None:
        colstoDefine = ["UserID", "ItemID", "Rewatch"]
    df = pd.read_csv(filepath_or_buffer=dfPath,
                     sep=",",
                     skiprows=1,
                     header=None,
                     dtype=dtype,
                     engine='python')
    df.columns = colstoDefine

    for idx, x in enumerate(divisionList):
        if idx + 1 < len(divisionList):
            y = divisionList[idx + 1]
            df_temp = df[(df[columnToDivide] > x) & (df[columnToDivide] < y)]
            save(df_temp, name + str(x) + '_' + str(y))
        else:
            df_temp = df[df[columnToDivide] > x]
            save(df_temp, name + str(x))


def oneHotEncoder(colstoOneHot, dfPath='../data/data_ICM_type.csv', colsToDelete=None,
                  colstoDefine=None, name=""):
    if colstoDefine is None:
        colstoDefine = ["ItemID", "Type", "Data"]
    df = pd.read_csv(filepath_or_buffer=dfPath,
                     sep=",",
                     skiprows=1,
                     header=None,
                     dtype={0: int, 1: int, 2: int},
                     engine='python')
    df.columns = colstoDefine

    if colsToDelete != None:
        for c in colsToDelete:
            df = delete_column(df, c)

    for c in colstoOneHot:
        df1 = pd.get_dummies(df[c], prefix=c)
        df = pd.concat([df, df1], axis=1)
        df = delete_column(df, c)

    save(df, name + "_1Hot")

def only_read_train_csr(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                            stats=False, preprocess=0, display=False, switch=False, dictionary=None, saving=False):
    n_items = 0

    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: str, 3: int},
                            engine='python')
    matrix_df.columns = columns

    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0})
    matrix = sps.coo_matrix((matrix_df[columns[3]].values,
                             (matrix_df[columns[0]].values, matrix_df[columns[1]].values)
                             ))
    if matrix_format == "csr":
        return matrix.tocsr()
    elif matrix_format == "csc":
        return matrix.tocsc()
    else:
        return matrix_df

def stacker(URM=None,ICM=None):
    #READING
    if URM==None:
        URM = only_read_train_csr(matrix_path="data/interactions_and_impressions.csv",matrix_format="...")
    if ICM==None:
        ICM = read_ICM_type(matrix_path="data/data_ICM_type.csv",matrix_format="...")
    n_users, n_items , n_features= factorization(URM,ICM)
    import numpy as np
    import scipy.sparse as sps

    #CSR CREATION
    URM_csr = sps.csr_matrix((URM["Data"].values,
                              (URM["UserID"].values, URM["ItemID"].values)),
                             shape = (n_users, n_items)) #always support a desired shape


    ICM_csr = sps.csr_matrix((np.ones(len(ICM["ItemID"].values)),
                              (ICM["ItemID"].values, ICM["FeatureID"].values)),
                             shape = (n_items, n_features))

    ICM_csr.data = np.ones_like(ICM_csr.data) #transfor array with all 1s if xisting val


    # STACKING
    # N_User * N_Item
    stacked_URM = sps.vstack([URM_csr, ICM_csr.T])
    stacked_URM = sps.csr_matrix(stacked_URM)

    # N_item * N_User
    stacked_ICM = sps.csr_matrix(stacked_URM.T)
    return stacked_URM,stacked_ICM

def read_train_csr(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                   stats=False, preprocess=0, display=False, switch=False, dictionary=None, column=None, saving=False):
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

    # df_col_normalize(matrix_df,columns[3],{0: 1, 1: 0})
    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0})
    if switch:
        df_col_normalize(matrix_df, colToChange=column, valsToPlace=dictionary)

    if preprocess > 0:
        print("Preprocessing with mode: " + str(preprocess))
        df_preprocess(matrix_df, saving=saving, mode=preprocess)

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
    # if is 0 (just description read) is less important

    if matrix_format == "csr":
        if stats:
            csr_stats(matrix.tocsr(), n_items)
        return matrix.tocsr()
    else:
        return matrix.tocsc()

    # colsToChange: The column where replace values
    # valsToPlace: dictionary of vals of kind:
    #  {oldval1: newval1 , oldval2: newval1,...}


def factorization(URM_all_dataframe, ICM_dataframe, enabled_userid=False):
    mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())

    print("Unique UserID in the URM are {}".format(len(original_id)))

    # total of all item in both
    if enabled_userid:
        all_item_indices = pd.concat([URM_all_dataframe["UserID"], ICM_dataframe["UserID"]], ignore_index=True)
    else:
        all_item_indices = pd.concat([URM_all_dataframe["UserID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique UserID in the URM and ICM are {}".format(len(original_id)))

    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    # ITEMS
    mapped_id, original_id = pd.factorize(URM_all_dataframe["ItemID"].unique())

    print("Unique ItemID in the URM are {}".format(len(original_id)))

    all_item_indices = pd.concat([URM_all_dataframe["ItemID"], ICM_dataframe["ItemID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique ItemID in the URM and ICM are {}".format(len(original_id)))

    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    mapped_id, original_id = pd.factorize(ICM_dataframe["FeatureID"].unique())
    feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    # MAPPING
    URM_all_dataframe["UserID"] = URM_all_dataframe["UserID"].map(user_original_ID_to_index)
    URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)

    print("Unique FeatureID in the URM are {}".format(len(feature_original_ID_to_index)))
    if enabled_userid:
        ICM_dataframe["UserID"] = ICM_dataframe["UserID"].map(user_original_ID_to_index)
    ICM_dataframe["ItemID"] = ICM_dataframe["ItemID"].map(item_original_ID_to_index)
    ICM_dataframe["FeatureID"] = ICM_dataframe["FeatureID"].map(feature_original_ID_to_index)
    n_users = len(user_original_ID_to_index)
    n_items = len(item_original_ID_to_index)
    n_features = len(feature_original_ID_to_index)
    return n_users, n_items , n_features


def df_preprocess(df, saving=True, mode=0):
    list_to_convert = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Passing through all dataset to gather infos..."):
        # print(index)
        # print(len(df))

        displayList = []
        if type(row[columns[2]]) == str:
            displayList = row[columns[2]].split(",")
            displayList = [eval(i) for i in displayList]

        userid = row[columns[0]]
        item = row[columns[1]]

        # 1-> Displayed (counts times a given Item has been displayed(in impressions list) to the user
        # 2-> Extended (adds an interaction (-1) if a User has an item in impression list
        # 3-> Count rewatches of each User-Item pair
        # 4-> Count all rewatches of each user
        if mode == 1:  # Displayed
            df.loc[index, 'Displayed'] = len(displayList)  # insert to the new column
        elif mode == 2:  # Extended
            for item in displayList:
                list_to_convert.append([userid, item, None, -1])
        elif mode == 3:  # Rewatch for each user-item
            if row[columns[3]] == 1:
                list_to_convert.append([userid, item])
        elif mode == 4:  # Rewatch (total) for each user or item
            if row[columns[3]] == 1:
                list_to_convert.append(item)

    if mode < 3:
        df1 = pd.DataFrame(list_to_convert, columns=columns)
        df = df.append(df1)
        df.columns = columns
        df = df.drop([2], axis=1)
        # df=df.sort_values(by=[0,1])
    elif mode == 3:
        cols = ["UserID", "ItemID", "Rewatch"]
        c = Counter(map(tuple, list_to_convert)).most_common()
        list_to_convert_final = []
        for i in tqdm(c, desc="Counting the rewatches"):
            list_to_convert_final.append([i[0][0], i[0][1], i[1]])
        df = pd.DataFrame(list_to_convert_final, columns=cols)
    elif mode == 4:
        cols = ["UserID", "Rewatch"]
        c = Counter(list_to_convert).most_common()
        df = pd.DataFrame(c, columns=cols)

    if saving:
        save(df, "out_" + str(mode))


def delete_column(df, colName):
    return df.drop([colName], axis=1)


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
    df.columns = ['ItemID', 'FeatureID', 'Episodes']

    # Since there's only one feature the FeatureID column is useless (all zeros)
    if clean:
        df = df.drop('feature', axis=1)
    #df.set_index('ItemID', inplace=True)
    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=df, columns=["EPLength"]).to_numpy())
    else:
        return sps.csc_matrix(pd.DataFrame(data=df, columns=["EPLength"]).to_numpy())


def read_ICM_type(matrix_path="../data/data_ICM_type.csv", matrix_format="csr", clean=True):
    df = pd.read_csv(filepath_or_buffer=matrix_path,
                     sep=",",
                     skiprows=1,
                     header=None,
                     dtype={0: int, 1: int, 2: int},
                     engine='python')
    df.columns = ['ItemID', 'FeatureID', 'data']

    # Since there's only one feature the data column is useless (all 1s)
    if clean:
        df = df.drop('data', axis=1)
    #df.set_index('ItemID', inplace=True)
    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())
    elif matrix_format == "csc":
        return sps.csc_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())
    else:
        return df



def get_user_segmentation(URM_train, URM_val, start_pos, end_pos):
    profile_length = np.ediff1d(URM_train.indptr)
    sorted_users = np.argsort(profile_length)
    users_in_group = sorted_users[start_pos:end_pos]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    return EvaluatorHoldout(URM_val, cutoff_list=[10], ignore_users=users_not_in_group)


def merge(ICM_a, ICM_b):
    return sps.hstack([ICM_a, ICM_b])


def save(data, name, relativePath="../output/", fullPath=None):
    if fullPath is None:
        data.to_csv(relativePath + name + '.csv', index=False)
    else:
        data.to_csv(fullPath, index=False)


def get_URM_ICM_Type(matrix_path_URM, matrix_path_ICM_type='../data_ICM_type.csv',
                     matrix_path_ICM_length='data/data_ICM_length.csv'):
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
                           index_col=False,
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

def get_URM_ICM_Type_Extended(matrix_path_URM, matrix_path_ICM_type='../data_ICM_type.csv',
                     matrix_path_ICM_length='data/data_ICM_length.csv'):
    columns = ["UserID", "ItemID","Data"]
    n_items = 0
    URM = pd.read_csv(filepath_or_buffer=matrix_path_URM,
                      sep=",",
                      skiprows=1,
                      header=None,
                      dtype={0: int, 1: int, 2: int},
                      engine='python')
    URM.columns = columns

    ICM_type = pd.read_csv(filepath_or_buffer=matrix_path_ICM_type,
                           index_col=False,
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
#    URM[columns[2]] = URM[columns[3]].replace({0: 1, 1: 0})
    # matrix_df.loc[~(matrix_df == 0).all(axis=2)]

    ICM_type["ItemID"] = ICM_type["ItemID"].map(item_original_ID_to_index)
    ICM_type["FeatureID"] = ICM_type["FeatureID"].map(feature_original_ID_to_index)

    n_users = len(user_original_ID_to_index)
    n_items = len(item_original_ID_to_index)
    n_features = len(feature_original_ID_to_index)



    URM_all = sps.csr_matrix((URM[columns[2]].values,
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


#################






if __name__ == '__main__':
    read_train_csr(preprocess=3)
