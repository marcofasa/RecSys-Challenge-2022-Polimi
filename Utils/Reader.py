from collections import Counter
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


def stacker(URM_train, ICM_all):
    stacked_URM = sps.vstack([URM_train, ICM_all.T])
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
    if vals_to_not_keep is not None:
        for val in vals_to_not_keep:
            data = data[data['Data'] != val]
    if values_to_replace is not None:
        data['Data'] = data['Data'].replace(values_to_replace)

    user_list = data['UserID'].tolist()
    item_list = data['ItemID'].tolist()
    rating_list = data['Data'].tolist()
    if matrix_format == "csr":
        return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()
    else:
        return data


def split_train_validation_double(URM_path="../data/interactions_and_impressions.csv", train_percentage=0.8,
                                  URM_path2="../data/interactions_and_impressions_v4.csv", vals_to_not_keep=None,
                                  values_to_replace=None, URM_cols=None, URM2_cols=None, URM_dtye=None, URM2_dtype=None,
                                  matrix_format="csr", URM=None, URM2=None):
    if URM_dtye is None:
        URM_dtye = {0: int, 1: int, 2: str, 3: int}
    if URM2_dtype is None:
        URM2_dtype = {0: int, 1: int, 3: float}

    if URM is not None:
        URM_all_dataframe = URM
    else:
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                        sep=",",
                                        skiprows=1,
                                        header=None,
                                        dtype=URM_dtye,
                                        engine='python')

    if URM_cols is not None:
        URM_all_dataframe.columns = URM_cols
    else:
        URM_all_dataframe.columns = ["UserID", "ItemID", "others", "Data"]
        URM_all_dataframe = URM_all_dataframe.drop("others", axis=1)
        URM_all_dataframe["Data"] = URM_all_dataframe["Data"].replace({0: 1, 1: 0})

    if URM2 is not None:
        URM_all_dataframe2 = URM2
    else:
        URM_all_dataframe2 = pd.read_csv(filepath_or_buffer=URM_path2,
                                         dtype=URM2_dtype)

    if URM2_cols is not None:
        URM_all_dataframe2.columns = URM2_cols
    else:
        URM_all_dataframe2.columns = ["UserID", "ItemID", "Data"]

    URM_all_dataframe2 = URM_all_dataframe2.sort_values(by=['UserID', 'ItemID', 'Data'])
    URM_all_dataframe = URM_all_dataframe.sort_values(by=['UserID', 'ItemID', 'Data'])

    if vals_to_not_keep is not None:
        for val in vals_to_not_keep:
            URM_all_dataframe2 = URM_all_dataframe2[URM_all_dataframe2['Data'] != val]
            URM_all_dataframe = URM_all_dataframe[URM_all_dataframe['Data'] != val]
    if values_to_replace is not None:
        URM_all_dataframe2['Data'] = URM_all_dataframe2['Data'].replace(values_to_replace)
        URM_all_dataframe['Data'] = URM_all_dataframe['Data'].replace(values_to_replace)

    all_users_items = list(set(list(zip(URM_all_dataframe['UserID'], URM_all_dataframe['ItemID']))))
    np.random.shuffle(all_users_items)

    import random

    user_for_training = random.sample(all_users_items, round(len(all_users_items) * train_percentage))
    user_for_validation = list(set(all_users_items) - set(user_for_training))
    user_for_training = pd.DataFrame(user_for_training, columns=["UserID", "ItemID"])
    user_for_validation = pd.DataFrame(user_for_validation, columns=["UserID", "ItemID"])
    data_train2 = URM_all_dataframe2[URM_all_dataframe2.set_index(["UserID", "ItemID"]).index.isin(
        user_for_training.set_index(["UserID", "ItemID"]).index)]
    data_train = URM_all_dataframe[URM_all_dataframe.set_index(["UserID", "ItemID"]).index.isin(
        user_for_training.set_index(["UserID", "ItemID"]).index)]
    # https://stackoverflow.com/questions/54006298/select-rows-of-a-dataframe-based-on-another-dataframe-in-python

    data_valid2 = URM_all_dataframe2[URM_all_dataframe2.set_index(["UserID", "ItemID"]).index.isin(
        user_for_validation.set_index(["UserID", "ItemID"]).index)]
    data_valid = URM_all_dataframe[URM_all_dataframe.set_index(["UserID", "ItemID"]).index.isin(
        user_for_validation.set_index(["UserID", "ItemID"]).index)]

    import scipy.sparse as sps

    URM_train = sps.csr_matrix((data_train["Data"].values,
                                (data_train["UserID"].values, data_train["ItemID"].values)))
    URM_train2 = sps.csr_matrix((data_train2["Data"].values,
                                 (data_train2["UserID"].values, data_train2["ItemID"].values)))
    URM_valid = sps.csr_matrix((data_valid["Data"].values,
                                (data_valid["UserID"].values, data_valid["ItemID"].values)))
    URM_valid2 = sps.csr_matrix((data_valid2["Data"].values,
                                 (data_valid2["UserID"].values, data_valid2["ItemID"].values)))
    if matrix_format == "csr":
        return URM_train, URM_train2, URM_valid, URM_valid2
    elif matrix_format == "df":
        return URM_all_dataframe, URM_all_dataframe2
    elif matrix_format == "total":
        return URM_all_dataframe, URM_all_dataframe2, URM_train, URM_train2, URM_valid, URM_valid2


def read_train_csr(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                   stats=False, preprocess=0, display=False, switch=False, dictionary=None, column=None, saving=False,
                   clean=False, values_to_replace=None, threshold=0):
    n_items = 0
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: str, 3: int},
                            engine='python')
    matrix_df.columns = columns
    # basic flipping
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
        df_preprocess(matrix_df, saving=saving, mode=preprocess, threshold=threshold)

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


def load_URM_and_ICM_items(URM_path="../data/interactions_and_impressions.csv", ICM_path="../data_ICM_type.csv",
                           ICM_cols=None, URM_cols=None,
                           ICM_dtype=None, URM_dtye=None, matrix_format="csr", URM=None, ICM_vals_to_not_keep=None,
                           URM_vals_to_not_keep=None, URM_values_to_replace=None, ICM_values_to_replace=None):
    if URM_dtye is None:
        URM_dtye = {0: int, 1: int, 2: str, 3: int}
    if ICM_dtype is None:
        ICM_dtype = {0: int, 1: int, 2: int}
    if URM is not None:
        URM_all_dataframe = URM
    else:
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                        sep=",",
                                        skiprows=1,
                                        header=None,
                                        dtype=URM_dtye,
                                        engine='python')

    if URM_cols is not None:
        URM_all_dataframe.columns = URM_cols
    else:
        URM_all_dataframe.columns = ["UserID", "ItemID", "others", "Interaction"]
    URM_all_dataframe["Interaction"] = URM_all_dataframe["Interaction"].replace({0: 1, 1: 0})

    if URM_vals_to_not_keep is not None:
        for val in URM_vals_to_not_keep:
            URM_all_dataframe = URM_all_dataframe[URM_all_dataframe['Interaction'] != val]
    if URM_values_to_replace is not None:
        URM_all_dataframe["Interaction"] = URM_all_dataframe["Interaction"].replace(URM_values_to_replace)
    ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path)

    ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path)
    if ICM_cols is not None:
        ICM_dataframe.columns = ICM_cols
    else:
        ICM_dataframe.columns = ["ItemID", "FeatureID", "Data"]

    if ICM_vals_to_not_keep is not None:
        for val in ICM_vals_to_not_keep:
            ICM_dataframe = ICM_dataframe[ICM_dataframe['FeatureID'] != val]

    if ICM_values_to_replace is not None:
        ICM_dataframe["FeatureID"] = ICM_dataframe["FeatureID"].replace(ICM_values_to_replace)

    ICM_dataframe = ICM_dataframe[ICM_dataframe["FeatureID"].notna()]
    n_features = len(ICM_dataframe["FeatureID"].unique())

    print("Number of tags\t {}, Number of item-tag tuples {}".format(n_features, len(ICM_dataframe)))
    mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())

    print("Unique UserID in the URM are {}".format(len(original_id)))

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
    URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)
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

    ICM_all.data = np.ones_like(ICM_all.data)
    if matrix_format == "csr":
        return URM_all, ICM_all
    elif matrix_format == "ICM":
        return URM_all_dataframe, ICM_all
    elif matrix_format == "URM":
        return URM_all, ICM_dataframe
    else:
        return URM_all_dataframe, ICM_dataframe


def load_URM_and_ICM_users(URM_path="../data/interactions_and_impressions.csv", ICM_path="../data/rewatches.csv",
                           ICM_cols=None, URM_cols=None, ICM_dtype=None, URM_dtye=None, matrix_format="csr",
                           ICM_vals_to_not_keep=None, URM_vals_to_not_keep=None, URM_values_to_replace=None,
                           ICM_values_to_replace=None):
    if URM_dtye is None:
        URM_dtye = {0: int, 1: int, 2: str, 3: int}
    if ICM_dtype is None:
        ICM_dtype = {0: int, 1: int, 2: int}
    import scipy.sparse as sps
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                    sep=",",
                                    skiprows=1,
                                    header=None,
                                    dtype=URM_dtye,
                                    engine='python')

    if URM_cols is not None:
        URM_all_dataframe.columns = URM_cols
    else:
        URM_all_dataframe.columns = ["UserID", "ItemID", "others", "Interaction"]
        URM_all_dataframe["Interaction"] = URM_all_dataframe["Interaction"].replace({0: 1, 1: 0})

    if URM_vals_to_not_keep is not None:
        for val in URM_vals_to_not_keep:
            URM_all_dataframe = URM_all_dataframe[URM_all_dataframe['Interaction'] != val]
    if URM_values_to_replace is not None:
        URM_all_dataframe["Interaction"] = URM_all_dataframe["Interaction"].replace(URM_values_to_replace)
    ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path)

    if ICM_cols is not None:
        ICM_dataframe.columns = ICM_cols
    else:
        ICM_dataframe.columns = ["UserID", "ItemID", "FeatureID"]

    if ICM_vals_to_not_keep is not None:
        for val in ICM_vals_to_not_keep:
            ICM_dataframe = ICM_dataframe[ICM_dataframe['FeatureID'] != val]

    if ICM_values_to_replace is not None:
        ICM_dataframe["FeatureID"] = ICM_dataframe["FeatureID"].replace(ICM_values_to_replace)
    ICM_dataframe = ICM_dataframe[ICM_dataframe["FeatureID"].notna()]
    n_features = len(ICM_dataframe["FeatureID"].unique())

    print("Number of tags\t {}, Number of item-tag tuples {}".format(n_features, len(ICM_dataframe)))
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

    ICM_all.data = np.ones_like(ICM_all.data)
    if matrix_format == "csr":
        return URM_all, ICM_all
    else:
        return URM_all_dataframe, ICM_dataframe


def read_train_csr_extended(matrix_path="../data/interactions_and_impressions.csv", matrix_format="csr",
                            stats=False, preprocess=0, display=False, switch=False, dictionary=None, column=None,
                            saving=False, replace=None, threshold=0):
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
        df_preprocess(matrix_df, saving=saving, mode=preprocess, threshold=threshold)

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

    if matrix_format == "csr":
        if stats:
            csr_stats(matrix.tocsr(), n_items)
        return matrix.tocsr()
    else:
        return matrix.tocsc()


def df_preprocess(df, saving=True, mode=0, threshold=0):
    list_to_convert = []
    list_to_check001 = set()
    list_to_check02 = set()
    list_to_check = []
    list_to_convert001 = []
    list_to_convert1 = ()
    cont = 0
    list_to_convert02 = []
    d = {}
    df = df.sort_values(by=['UserID', 'ItemID', 'Data'])
    userid = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Passing through all dataset to gather infos..."):
        # print(index)
        # print(len(df))

        displayList = []
        if type(row[columns[2]]) == str:
            displayList = row[columns[2]].split(",")
            displayList = [eval(i) for i in displayList]
        if mode < 13 and userid != row[columns[0]]:
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

        elif mode == 10:  # v4-> also 0.1
            cont = cont + 1
            for i in displayList:
                if i != item:
                    list_to_convert001.append([userid, i, None, 0.01])
        elif mode == 11:
            if row[columns[3]] == 1:
                if [userid, item, 0.2] in list_to_convert02:
                    # list_to_convert.remove([userid, item, 0.2])
                    for _ in list(filter(lambda a: a == [userid, item, 0.2], list_to_convert02)):
                        list_to_convert.append([userid, item, 0.5])
                    list_to_convert.append([userid, item, 1])

                    list_to_convert02 = list(filter(lambda a: a != [userid, item, 0.2], list_to_convert02))

                elif [userid, item, 0.01] in list_to_convert001:
                    # list_to_convert.remove([userid, item, 0.01])
                    for _ in list(filter(lambda a: a == [userid, item, 0.01], list_to_convert001)):
                        list_to_convert.append([userid, item, 0.05])
                    list_to_convert.append([userid, item, 1])

                    list_to_convert001 = list(filter(lambda a: a != [userid, item, 0.01], list_to_convert001))
                else:
                    list_to_convert.append([userid, item, 1])  # provare con 0.9
            elif row[columns[3]] == 0:
                list_to_convert02.append([userid, item, 0.2])
            for i in displayList:
                if i != item:
                    list_to_convert001.append([userid, i, 0.01])
        if mode == 12:
            if row[columns[3]] == 1:
                list_to_convert1.append((userid, item, 1))
            else:
                list_to_convert02.append((userid, item, 0.2))

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
    elif mode == 6 or mode == 7 or mode == 8 or mode == 9 or mode == 11:
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
    elif mode == 12:
        for i in range(len(list_to_convert1) - 1):
            x = list_to_convert1[i]
            c = 0
            for j in range(i, len(list_to_convert1)):
                if list_to_convert1[j] == list_to_convert1[i]:
                    c = c + 1
            count = dict({x: c})
            if x not in d.keys():
                d.update(count)
        more_than = {k: v for k, v in d.items() if v >= threshold}
        list(more_than.keys())

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


def load_ICM_type(file_path="../data/data_ICM_type.csv", item_icm_col="item_id", feature_icm_col="feature_id",
                  weight_icm_col="data"):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata[item_icm_col].tolist()
    feature_list = metadata[feature_icm_col].tolist()
    weight_list = metadata[weight_icm_col].tolist()
    ICM_all = sps.csr_matrix((np.ones(len(item_icm_list)),
                              (item_icm_list, feature_list)),
                             )

    ICM_all.data = np.ones_like(ICM_all.data)  # transfor array with all 1s if xisting val

    return ICM_all


def load_ICM_rewatches_total(file_path="../data/rewatches/rewatches_total.csv", item_icm_col="item_id",
                             feature_icm_col="rewatches", weight_icm_col="data"):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata[item_icm_col].tolist()
    feature_list = metadata[feature_icm_col].tolist()
    ICM_all = sps.csr_matrix((np.ones(len(item_icm_list)),
                              (item_icm_list, feature_list)),
                             )

    ICM_all.data = np.ones_like(ICM_all.data)  # transfor array with all 1s if xisting val

    return ICM_all


#################


if __name__ == '__main__':
    read_train_csr(preprocess=12, saving=True,threshold=100)
