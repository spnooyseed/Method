import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import torch
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


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)


class DataDiffusion2(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        return item1, item2

    def __len__(self):
        return len(self.data1)


class DataDiffusion3(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]
        return item1, item2, item3

    def __len__(self):
        return len(self.data1)


def get_top_k_similar_pearson(data, k):
    # Subtract the mean of each row from the rows (center the data)
    mean_centered_data = data - data.mean(dim=1, keepdim=True)

    # Compute the covariance matrix
    covariance_matrix = torch.mm(mean_centered_data, mean_centered_data.t())

    # Normalize the covariance matrix to get Pearson correlation coefficients
    # Calculate the standard deviation for each row
    std_dev = mean_centered_data.norm(p=2, dim=1, keepdim=True)

    # Avoid division by zero in case there is a row with zero variance
    std_dev[std_dev == 0] = 1

    # Pearson correlation matrix
    pearson_correlation_matrix = covariance_matrix / torch.mm(std_dev, std_dev.t())

    # We need to zero out the diagonal elements (self-correlation) before getting top-k
    # Fill diagonal with very low value which cannot be a top correlation
    eye = torch.eye(
        pearson_correlation_matrix.size(0), device=pearson_correlation_matrix.device
    )
    pearson_correlation_matrix -= (
        eye * 2
    )  # Subtract 2 which is definitely out of bound for correlation

    # Get top-k values along each row
    _, indices = torch.topk(pearson_correlation_matrix, k=k, dim=1)

    return indices
