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
    data = stocks_group['high'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)  # fill the missing values
    dates = stocks_group['date'].apply(lambda x: pd.Series(x.values)).unstack()
    train_ind, test_ind = create_indices(len(data.values), 0.8)
    data_values = np.asarray(data.values)
    train_data = np.asarray(np.array_split(data_values[train_ind], 19, axis=1)).transpose((1, 0, 2))  #split into n sub-sequences
    test_data = np.asarray(np.array_split(data_values[test_ind], 19, axis=1)).transpose((1, 0, 2))     #split into n sub-sequences
    mean_train, std_train = normalize(train_data)
    mean_test, std_test = normalize(test_data)

    return {'dates:': np.asarray(dates[:1]).flatten(),
            'train_data': torch.FloatTensor(train_data),
            'train_mean': mean_train,
            'train_std': std_train,
            'test_data': torch.FloatTensor(test_data),
            'test_mean': mean_test,
            'test_std': std_test,
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

set_seed(1)
split_data(parse())