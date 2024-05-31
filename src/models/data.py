#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   data.py
"""
from datetime import timedelta

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as reset
import model
import time
import numpy as np


def get_dataLoad(config, path):
    df = pd.read_csv(path, header=None)
    train_data, test_data = train_test_split(df)
    traindataset, testdataset = MyDataset(train_data), MyDataset(test_data)
    train, test = DataLoader(traindataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                             drop_last=True), DataLoader(testdataset, batch_size=config.batch_size, shuffle=False,
                                                         num_workers=0, drop_last=True)
    end_time = time.time()
    return train, test


def get_dataLoad_test(config, path):
    df = pd.read_csv(path, header=None)
    test_data = df
    testdataset = MyDataset(test_data)
    test = DataLoader(testdataset, batch_size=config.batch_size, shuffle=False,
                      num_workers=0, drop_last=True)
    end_time = time.time()
    return test


def data_pre(data):
    x = []
    for item in data:
        nums = item.split(',')
        x1, x2, x3, label = nums[:256], nums[256:512], nums[512:-1], nums[-1]

        x.append(x1 + x2 + label)

    x_new = torch.tensor(x)
    df = pd.DataFrame(x_new)
    return df


def train_test_split(data, train_size=0.8, shuffle=True, random_state=42):
    if shuffle:
        data = reset(data)

    train = data[:int(len(data) * train_size)].reset_index(drop=True)
    test = data[int(len(data) * train_size):].reset_index(drop=True)
    return train, test


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.data1 = data.iloc[:, :256]
        # self.data2 = data.iloc[:, 256:512]
        # self.data3 = data.iloc[:, 512:-1]
        # self.label = data.iloc[:, -1]

    def __getitem__(self, index):
        line = self.data.iloc[index]
        x1 = torch.tensor(line[:256].to_numpy(float), dtype=torch.float)
        x2 = torch.tensor(line[256:512].to_numpy(float), dtype=torch.float)
        x3 = torch.tensor(line[512:-1].to_numpy(float), dtype=torch.float)
        label = torch.tensor(line.iloc[-1], dtype=torch.float)
        return x1, x2, x3, label
        # x1 = torch.tensor(line[:128].to_numpy(float), dtype=torch.float)
        # x2 = torch.tensor(line[128:256].to_numpy(float), dtype=torch.float)
        # label = torch.tensor(line.iloc[-1], dtype=torch.float)
        # return x1, x2, label

    def __len__(self):
        return len(self.data)
