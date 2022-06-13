import pandas as pd
import os
import numpy as np
import torch


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse():
    stocks = pd.read_csv(f"{os.getcwd()}/SP 500 Stock Prices 2014-2017.csv")
    return stocks.sort_values(by='date')


def split_data(stocks):
    stocks = stocks[["symbol", "high", "date"]]
    stocks_group = stocks.groupby('symbol')
    stocks_symbol = np.asarray(stocks[["symbol"]].drop_duplicates())
    data = stocks_group['high'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)  # fill the missing values
    dates = stocks_group['date'].apply(lambda x: pd.Series(x.values)).unstack()
    train_ind, test_ind = create_indices(len(data.values), 0.8)
    data_values = np.asarray(data.values)
    train_data = np.asarray(np.array_split(data_values[train_ind], 19, axis=1)).transpose((1, 0, 2))  #split into n sub-sequences
    test_data = np.asarray(np.array_split(data_values[test_ind], 19, axis=1)).transpose((1, 0, 2))     #split into n sub-sequences
    mean_train, std_train = normalize(train_data)
    mean_test, std_test = normalize(test_data)
    stocks_train_name = stocks_symbol[train_ind]
    stocks_test_name = stocks_symbol[test_ind]

    return {'dates': np.asarray(dates[:1]).flatten(),
            'train_set': torch.FloatTensor(train_data),
            'train_mean': mean_train,
            'train_std': std_train,
            'train_name': stocks_train_name,
            'test_set': torch.FloatTensor(test_data),
            'test_mean': mean_test,
            'test_std': std_test,
            'test_name': stocks_test_name
            }


def create_indices(n, train_portion):
    indices = np.random.permutation(n)
    train_size = int(n * train_portion)
    return indices[: train_size], indices[train_size:]


def normalize(data):
    mean_data = []
    std_data = []
    for i, d in enumerate(data):   #go over each symbol line
        curr_mean = []
        curr_std = []
        for j, d_seq in enumerate(d):       #go over each sub sequence
            mean = np.mean(d_seq)
            std = np.std(d_seq)
            data[i][j] = (d_seq - mean) / std

            curr_mean.append(mean)
            curr_std.append(std)
        mean_data.append(curr_mean)
        std_data.append(curr_std)
    return mean_data, std_data


def revert_normalize(data, data_mean, data_std):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j] * data_std[i][j] + data_mean[i][j]


set_seed(1)
split_data(parse())