import os
import pickle
import numpy as np
import scipy.sparse as sp


def process_file(dataset, file_name, save_name):
    file = os.path.join("./Datasets", dataset, file_name)
    with open(file, "rb") as fs:
        test_list = (pickle.load(fs) != 0).astype(np.float32)
    if type(test_list) is not sp.coo_matrix:
        test_list = sp.coo_matrix(test_list)
    user_item_array = np.column_stack((test_list.row, test_list.col))
    data_to_save = {"user_item_array": user_item_array, "shape": test_list.shape}
    save_path = os.path.join("./Datasets", dataset, save_name)
    np.save(save_path, data_to_save)


def process(dataset):
    process_file(dataset, "tstMat.pkl", "test_list.npy")
    process_file(dataset, "trnMat.pkl", "train_list.npy")


process("ml-1m")
process("douban-book")
process("yelp")
