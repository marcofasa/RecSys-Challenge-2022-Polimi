import pandas as pd
import scipy.sparse as sps
import numpy as np
URM_path = "/home/vittorio/Scrivania/Politecnico/RecSys/RecSys_DEPRECATED/Dataset/interactions_and_impressions.csv"
URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                sep=",",
                                header=None,
                               # dtype={0:int, 1:int, 2:str,4:int},
                                engine='python')  # its a way to store the data, they are sepatated by sep

URM_all_dataframe.columns = ["UserID", "ItemID", "Impression_list", "Data"]
print(URM_all_dataframe.head(n=10))



ICM_path="/home/vittorio/Scrivania/Politecnico/RecSys/RecSys_DEPRECATED/Dataset/data_ICM_type.csv"
ICM_dataframe = pd.read_csv(filepath_or_buffer=ICM_path,
                            sep=",",
                            header=None,
                            #dtype={0:int, 1:int, 2:str, 3:int},
                            engine='python')

ICM_dataframe.columns = ["UserID", "ItemID", "FeatureID"]
ICM_dataframe = ICM_dataframe[ICM_dataframe["FeatureID"].notna()]
print(ICM_dataframe.head(n=10))
n_features = len(ICM_dataframe["FeatureID"].unique())

print ("Number of tags\t {}, Number of item-tag tuples {}".format(n_features, len(ICM_dataframe)))
mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())

print("Unique UserID in the URM are {}".format(len(original_id)))


all_item_indices = pd.concat([URM_all_dataframe["ItemID"], ICM_dataframe["ItemID"]], ignore_index=True)
mapped_id, original_id = pd.factorize(all_item_indices.unique())

print("Unique ItemID in the URM and ICM are {}".format(len(original_id)))

item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(ICM_dataframe["FeatureID"].unique())
feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)

print("Unique FeatureID in the URM are {}".format(len(feature_original_ID_to_index)))

URM_all_dataframe["UserID"] = URM_all_dataframe["UserID"].map(user_original_ID_to_index)
URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)
print(URM_all_dataframe.head(n=10))

ICM_dataframe["UserID"] = ICM_dataframe["UserID"].map(user_original_ID_to_index)
ICM_dataframe["ItemID"] = ICM_dataframe["ItemID"].map(item_original_ID_to_index)
ICM_dataframe["FeatureID"] = ICM_dataframe["FeatureID"].map(feature_original_ID_to_index)
ICM_dataframe.head(n=10)


n_users = len(user_original_ID_to_index)
n_items = len(item_original_ID_to_index)
n_features = len(feature_original_ID_to_index)

#URM_all_dataframe["Data"][0]=0
URM_all = sps.csr_matrix((URM_all_dataframe["Data"].values,
                          (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)),
                        shape = (n_users, n_items))

print(URM_all)


ICM_all = sps.csr_matrix((np.ones(len(ICM_dataframe["ItemID"].values)),
                          (ICM_dataframe["ItemID"].values, ICM_dataframe["FeatureID"].values)),
                        shape = (n_items, n_features))

ICM_all.data = np.ones_like(ICM_all.data)



print(ICM_all)
