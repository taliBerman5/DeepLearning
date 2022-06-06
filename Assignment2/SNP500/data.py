import pandas as pd
import os
import numpy as np
import torch


def parse():
    stocks = pd.read_csv(f"{os.getcwd()}/SP 500 Stock Prices 2014-2017.csv")
    return stocks.sort_values(by='date')

def split_data(stocks):
    stocks = stocks[["symbol", "high", "date"]]
    stocks_group = stocks.groupby('symbol')
    data = stocks_group['high'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)
    dates = stocks_group['date'].apply(lambda x: pd.Series(x.values)).unstack()
    trainInd, testInd = create_indices(len(data.values), 0.8)
    dataValues = np.asarray(data.values)
    trainList = dataValues[trainInd]
    testList = dataValues[testInd]
    trainData = np.asarray(np.array_split(trainList, 19, axis=1)).transpose((1, 0, 2))
    # trainData = np.asarray(toNormal(trainList))
    testData = np.asarray(np.array_split(testList, 19, axis=1)).transpose((1, 0, 2))
    # testData = np.asarray(toNormal(testList))
    trainTensor = torch.FloatTensor(toNormal(trainData, False))
    testTensor = torch.FloatTensor(toNormal(testData, True))

    # trainTensor = np.array_split(trainTensor, numGroups)

    return trainTensor, testTensor, np.asarray(dates[:1]).flatten()



def create_indices(n, train_portion):
    indices = np.random.permutation(n)
    train_size = int(n * train_portion)
    return indices[: train_size], indices[train_size:]


