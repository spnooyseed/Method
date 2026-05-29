import numpy as np
import torch
import scipy.sparse as sp
import pickle


def data_load(train_path, test_path):
    # train_list = np.load(train_path, allow_pickle=True)
    with open(train_path, "rb") as fs:
        train_data = (pickle.load(fs) != 0).astype(np.float32)
    if type(train_data) is not sp.csr_matrix:
        train_data = sp.csr_matrix(train_data)
    with open(test_path, "rb") as fs:
        test_y_data = (pickle.load(fs) != 0).astype(np.float32)
    if type(test_y_data) is not sp.csr_matrix:
        test_y_data = sp.csr_matrix(test_y_data)

    n_user, n_item = train_data.shape
    print(f"user num: {n_user}")
    print(f"item num: {n_item}")

    return train_data, test_y_data, n_user, n_item


dataset_name = "yelp"
data_path = "/home/sunkaikai/workspace/GDiffMAE-main/Datasets/"
import os

train_path = os.path.join(data_path, dataset_name, "trnMat.pkl")
test_path = os.path.join(data_path, dataset_name, "tstMat.pkl")
print(train_path)
print(test_path)

train_data, test_y_data, n_user, n_item = data_load(train_path, test_path)
print(train_data.shape)
# print(train_data.nbytes)

data = train_data.todense()
print(data.shape)
print(data.nbytes)
#
print("ints:", np.sum(np.sum(data, axis=1)))


def get_2hop_item_based(data):
    # Initialize an empty tensor
    sec_hop_infos = torch.empty(len(data), len(data[0]))
    print(sec_hop_infos.size())

    # Loop to add data to the tensor
    sec_hop_inters = torch.sum(data, axis=0) / n_user
    for i, row in enumerate(data):

        zero_indices = torch.nonzero(row < 0.000001).t()  # .squeeze()
        if i % 1000 == 0:
            print(i)

        sec_hop_infos[i] = sec_hop_inters
        sec_hop_infos[i][zero_indices[0]] = 0

    # tensor = torch.cat((data, sec_hop_infos), dim=1)  # Concatenate the data to the tensor

    return sec_hop_infos


# Call the function
hop2_rates_test = get_2hop_item_based(torch.tensor(data, dtype=torch.float32))

# Print the resulting tensor
print("hop2_rates_test.size() = ", hop2_rates_test.size())

filename = os.path.join(data_path, dataset_name, "two_hop_rates_items.pt")
torch.save(hop2_rates_test, filename)
