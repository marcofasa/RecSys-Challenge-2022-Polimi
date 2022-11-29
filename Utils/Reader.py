import math
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sps
from matplotlib import pyplot
from tqdm import tqdm
import os

from Recommenders.Recommender_utils import check_matrix
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


def df_col_replace(df, col_to_change, values_to_change):
    df[col_to_change] = df[col_to_change].replace(values_to_change)
    return


'''
Structure of the ICM -> UserID,ItemID,FeatureID
ASSERT IT HAS 3 COLS
'''


def stacker(URM_path="../data/interactions_and_impressions.csv", ICM_path='../data/rewatches.csv',
            ICM_cols_to_drop=None, ICM_values_to_change=None):
    # CON MAPPING
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                    sep=",",
                                    skiprows=1,
                                    header=None,
                                    dtype={0: int, 1: int, 2: str, 3: int},
                                    engine='python')

    URM_all_dataframe.columns = ["UserID", "ItemID", "NULL", "Interaction"]
    URM_all_dataframe["Interaction"] = URM_all_dataframe["Interaction"].replace({0: 1, 1: 0})
    ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path,
                                sep=",",
                                skiprows=1,
                                header=None,
                                dtype={0: int, 1: int, 2: int},
                                engine='python')

    if (ICM_cols_to_drop != None):
        for col in ICM_cols_to_drop:
            ICM_dataframe = ICM_dataframe.drop(col, axis=1)
    if (ICM_values_to_change != None):
        df_col_replace(ICM_dataframe, 2, ICM_values_to_change)

    ICM_dataframe.columns = ["UserID", "ItemID", "FeatureID"]

    # Some nan values exist, remove them
    ICM_dataframe = ICM_dataframe[ICM_dataframe["FeatureID"].notna()]

    n_features = len(ICM_dataframe["FeatureID"].unique())

    print("Number of tags\t {}, Number of item-tag tuples {}".format(n_features, len(ICM_dataframe)))
    ## Build the sparse URM and ICM matrices

    mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())

    print("Unique UserID in the URM are {}".format(len(original_id)))

    all_item_indices = pd.concat([URM_all_dataframe["UserID"], ICM_dataframe["UserID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique UserID in the URM and ICM are {}".format(len(original_id)))

    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    mapped_id, original_id = pd.factorize(URM_all_dataframe["ItemID"].unique())

    print("Unique ItemID in the URM are {}".format(len(original_id)))

    all_item_indices = pd.concat([URM_all_dataframe["ItemID"], ICM_dataframe["ItemID"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_item_indices.unique())

    print("Unique ItemID in the URM and ICM are {}".format(len(original_id)))

    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    mapped_id, original_id = pd.factorize(ICM_dataframe["FeatureID"].unique())
    feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    print("Unique FeatureID in the URM are {}".format(len(feature_original_ID_to_index)))

    URM_all_dataframe["UserID"] = URM_all_dataframe["UserID"].map(user_original_ID_to_index)
    URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)
    ICM_dataframe["UserID"] = ICM_dataframe["UserID"].map(user_original_ID_to_index)
    ICM_dataframe["ItemID"] = ICM_dataframe["ItemID"].map(item_original_ID_to_index)
    ICM_dataframe["FeatureID"] = ICM_dataframe["FeatureID"].map(feature_original_ID_to_index)

    n_users = len(user_original_ID_to_index)
    n_items = len(item_original_ID_to_index)
    n_features = len(feature_original_ID_to_index)
    URM_all = sps.csr_matrix((URM_all_dataframe["Interaction"].values,
                              (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)),
                             shape=(n_users, n_items))  # always support a desired shape

    ICM_all = sps.csr_matrix((np.ones(len(ICM_dataframe["ItemID"].values)),
                              (ICM_dataframe["ItemID"].values, ICM_dataframe["FeatureID"].values)),
                             shape=(n_items, n_features))

    ICM_all.data = np.ones_like(ICM_all.data)  # transfor array with all 1s if xisting val

    stacked_URM = sps.vstack([URM_all, ICM_all.T])
    stacked_URM = sps.csr_matrix(stacked_URM)

    stacked_ICM = sps.csr_matrix(stacked_URM.T)

    return stacked_URM, stacked_ICM
'''
- vals_to_not_keep: Array of row to delete comparing value in Data

'''

def load_URM(file_path="../data/URM_new.csv", values_to_replace=None, vals_to_not_keep=None, matrix_format="csr"):
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

    data = pd.read_csv(file_path)

    import scipy.sparse as sps
    if values_to_replace is not None:
        data['Data'] = data['Data'].replace(values_to_replace)
    if vals_to_not_keep is not None:
        for val in vals_to_not_keep:
            data = data[data['Data'] != val]
    user_list = data['UserID'].tolist()
    item_list = data['ItemID'].tolist()
    rating_list = data['Data'].tolist()
    if matrix_format == "csr":
        return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()
    else:
        return data


def read_train_csr(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                   stats=False, preprocess=0, display=False, switch=False, dictionary=None, column=None, saving=False,
                   clean=False,values_to_replace=None):
    n_items = 0
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: str, 3: int},
                            engine='python')
    matrix_df.columns = columns
    #basic flipping
    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0})
    if values_to_replace is not None:
        matrix_df[columns[3]] = matrix_df[columns[3]].replace(values_to_replace)
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
    # TODO rimmetti come era prima il replace
    # matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0.04})
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
    if clean:
        matrix_df = matrix_df.drop_duplicates(subset='Data', keep=False)

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


def read_train_csr_extended(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                            stats=False, preprocess=0, display=False, switch=False, dictionary=None, column=None,
                            saving=False, replace=None):
    n_items = 0
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: float},
                            engine='python')
    matrix_df.columns = ["UserID", "ItemID", "Data"]
    if replace is not None:
        df_col_replace(matrix_df, col_to_change="Data", values_to_change=replace)
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
    matrix_df[columns[3]] = matrix_df[columns[3]].replace({0: 1, 1: 0, -1: -0.15})
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
    return n_users, n_items, n_features


def df_preprocess(df, saving=True, mode=0):
    list_to_convert = []
    list_to_check001 = set()
    list_to_check02 = set()
    list_to_check = []
    list_to_convert001 = []
    cont = 0
    list_to_convert02 = []

    df = df.sort_values(by=['UserID', 'ItemID', 'Data'])
    userid = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Passing through all dataset to gather infos..."):
        # print(index)
        # print(len(df))

        displayList = []
        if type(row[columns[2]]) == str:
            displayList = row[columns[2]].split(",")
            displayList = [eval(i) for i in displayList]
        if mode < 10 and userid != row[columns[0]]:
            userid = row[columns[0]]
            list_to_convert = list_to_convert + list_to_convert001
            list_to_convert = list_to_convert + list_to_convert02
            list_to_convert001.clear()
            list_to_convert02.clear()
        else:
            userid = row[columns[0]]

        item = row[columns[1]]
        # At this point you have userid,item and a displaylist with all items displayed in background

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
        elif mode == 4:  # Rewatch (total) for each user
            if row[columns[3]] == 1:
                list_to_convert.append(userid)
        elif mode == 5:  # Rewatch (total) for each user
            if row[columns[3]] == 1:
                list_to_convert.append(item)
        elif mode == 6:
            # with set you have no duplicates
            for i in displayList:
                list_to_convert.append([userid, i, 0.01])
                list_to_check001.add((userid, i))
            if row[columns[3]] == 1:
                if (userid, item) in list_to_check02:
                    # list_to_convert.remove([userid, item, 0.2])
                    list_to_convert.append([userid, item, 0.8])
                    list_to_check02.remove((userid, item))
                    list_to_convert.remove([userid, item, 0.2])
                elif (userid, item) in list_to_check001:
                    # list_to_convert.remove([userid, item, 0.01])
                    list_to_convert.append([userid, item, 0.5])
                    list_to_check001.remove((userid, item))
                    list_to_convert.remove([userid, item, 0.01])
                else:
                    list_to_convert.append([userid, item, 1])
            elif row[columns[3]] == 0:

                list_to_convert.append([userid, item, 0.2])
                list_to_check02.add((userid, item))
        elif mode == 7:
            # with duplicates--> Full exposition
            for i in displayList:
                if (userid, i) not in list_to_check001:
                    list_to_convert.append([userid, i, 0.01])
                    list_to_check001.add((userid, i))
            if row[columns[3]] == 1:
                if (userid, item) in list_to_check02:
                    # list_to_convert.remove([userid, item, 0.2])
                    list_to_convert.append([userid, item, 0.8])
                    list_to_check02.remove((userid, item))
                    list_to_convert.remove([userid, item, 0.2])
                elif (userid, item) in list_to_check001:
                    # list_to_convert.remove([userid, item, 0.01])
                    list_to_convert.append([userid, item, 0.5])
                    list_to_check001.remove((userid, item))
                    list_to_convert.remove([userid, item, 0.01])
                else:
                    list_to_convert.append([userid, item, 1])
            elif row[columns[3]] == 0:
                if (userid, item) not in list_to_check02:
                    list_to_convert.append([userid, item, 0.2])
                    list_to_check02.add((userid, item))
        elif mode == 8:
            if row[columns[3]] == 1:
                if [userid, item, 0.2] in list_to_convert02:
                    # list_to_convert.remove([userid, item, 0.2])
                    list_to_convert.append([userid, item, 0.8])
                    list_to_convert.append([userid, item, 1])
                    list_to_convert02.remove([userid, item, 0.2])

                elif [userid, item, 0.01] in list_to_convert001:
                    # list_to_convert.remove([userid, item, 0.01])
                    list_to_convert.append([userid, item, 0.5])
                    list_to_convert.append([userid, item, 1])
                    list_to_convert001.remove([userid, item, 0.01])
                else:
                    list_to_convert.append([userid, item, 1])
            elif row[columns[3]] == 0:
                list_to_convert02.append([userid, item, 0.2])
            for i in displayList:
                if i != item:
                    list_to_convert001.append([userid, i, 0.01])
        elif mode == 9:
            if row[columns[3]] == 1:
                list_to_convert.append([userid, item, 1])
            elif row[columns[3]] == 0:
                list_to_convert02.append([userid, item, 0.2])
            for i in displayList:
                if i != item:
                    list_to_convert001.append([userid, i, 0.01])
        elif mode == 10:
            cont = cont + 1
            for i in displayList:
                if i != item:
                    list_to_convert001.append([userid, i, None, 0.01])

        if cont == 10:
            break
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
    elif mode == 5:
        cols = ["ItemID", "Rewatch"]
        c = Counter(list_to_convert).most_common()
        df = pd.DataFrame(c, columns=cols)
    elif mode == 6 | mode == 7 | mode == 8 | mode == 9:
        cols = ["UserID", "ItemID", "Data"]
        # list_to_convert = list(dict.fromkeys(list_to_convert))  # removing duplicates
        del df
        df = pd.DataFrame(list_to_convert, columns=cols)
    elif mode == 10:
        df1 = pd.DataFrame(list_to_convert001, columns=columns)
        df = df.append(df1)
        df.columns = columns
        df = df.drop(["Interaction"], axis=1)
        df = df.sort_values(by=['UserID', 'ItemID', 'Data'])

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


def read_ICM_rewatches(matrix_path="../data/rewatches.csv", matrix_format="csr", clean=True):
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
    # df.set_index('ItemID', inplace=True)
    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())
    elif matrix_format == "csc":
        return sps.csc_matrix(pd.DataFrame(data=df, columns=["Type"]).to_numpy())
    else:
        return df


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
    # df.set_index('ItemID', inplace=True)
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
    # df.set_index('ItemID', inplace=True)
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


def get_URM_ICM_Type(matrix_path_URM, matrix_path_ICM_type='../data/data_ICM_type.csv',
                     matrix_path_ICM_length='../data/data_ICM_length.csv'):
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
    columns = ["UserID", "ItemID", "Data"]
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


def combine(ICM: sps.csr_matrix, URM: sps.csr_matrix):
    return sps.hstack((URM.T, ICM), format='csr')


def binarize(x):
    if x != 0:
        return 1
    return x


def binarize_ICM(ICM: sps.csr_matrix):
    vbinarize = np.vectorize(binarize)

    ICM.data = vbinarize(ICM.data)


def linear_scaling_confidence(URM_train, alpha):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * C.data

    return C


def load_ICM_rewatches(file_path='../data/rewatches.csv'):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata['ItemID'].tolist()
    feature_list = metadata['UserID'].tolist()
    weight_list = metadata['Rewatch'].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))


def load_ICM_displayed(file_path='../data/displayed.csv', weight_list_col='Displayed'):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata['ItemID'].tolist()
    feature_list = metadata['UserID'].tolist()
    weight_list = metadata[weight_list_col].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))


'''
Works with both data_ICM_type and data_ICM_length
'''


def load_ICM(file_path, item_icm_col="item_id", feature_icm_col="feature_id", weight_icm_col="data"):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata[item_icm_col].tolist()
    feature_list = metadata[feature_icm_col].tolist()
    weight_list = metadata[weight_icm_col].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))


#################


if __name__ == '__main__':
    read_train_csr(preprocess=8, saving=True)
