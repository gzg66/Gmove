#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   models.py
"""

import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV


# 定义一个config类，用于初始化参数
class config(object):
    # 初始化参数
    def __init__(self, model_name):
        # 模型名称
        self.model_name = model_name
        # 保存路径
        self.save_path = 'E:\gzg\PycharmProjects\PRMove\data\\result' + self.model_name + 'RQ3_3.ckpt'
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 要求改进
        self.require_improvment = 1000
        # 类别数
        self.num_class = 3
        # epoch数
        self.num_epochs = 120
        # 批次大小
        # self.batch_size = 128
        self.batch_size = 256
        # self.batch_size = 1
        # 填充大小
        self.padding_size = 20
        # 学习率
        self.learning_rate = 1e-4
        # GRU隐藏层
        self.gru_hidden = 256
        # RNN隐藏层
        self.rnn_hidden = 256
        # 隐藏层
        # self.hidden_size = 128
        self.hidden_size = 256
        # 过滤器
        self.num_filter = 256
        # 池化类型
        self.pool_type = max
        # 层数
        self.num_layer = 2
        # dropout
        self.dropout = 0.5
        # 线性
        self.linear = 128


# 定义CNN
class CNN_METER(nn.Module):
    def __init__(self):
        super(CNN_METER, self).__init__()
        self.cnn = nn.Sequential(
            torch.nn.Conv1d(81, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
        )
        # 定义展平层
        self.flatten = nn.Flatten()
        # 定义全连接层
        # self.fc = torch.nn.Linear(128, 256)

    def forward(self, inputs):
        # 将输入的形状转换为(batch_size,channels,length)
        inputs = inputs.view(inputs.shape[0], -1, 1)
        # 运行CNN模型
        out = self.cnn(inputs)
        # 将CNN模型的输出展平
        out = self.flatten(out)
        # 将展平后的输出运行全连接层
        # out = self.fc(out)
        # 返回输出
        return out


class CNN_GE(nn.Module):
    def __init__(self):
        super(CNN_GE, self).__init__()
        self.cnn = nn.Sequential(
            torch.nn.Conv1d(256, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
        )
        # 定义展平层
        self.flatten = nn.Flatten()
        # 定义全连接层
        # self.fc = torch.nn.Linear(128, 256)

    def forward(self, inputs):
        # 将输入的形状转换为(batch_size,channels,length)
        inputs = inputs.view(inputs.shape[0], -1, 1)
        # 运行CNN模型
        out = self.cnn(inputs)
        # 将CNN模型的输出展平
        out = self.flatten(out)
        # 将展平后的输出运行全连接层
        # out = self.fc(out)
        # 返回输出
        return out


class CNN_ALL(nn.Module):
    def __init__(self):
        super(CNN_ALL, self).__init__()
        self.cnn = nn.Sequential(
            torch.nn.Conv1d(384, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
        )
        # 定义展平层
        self.flatten = nn.Flatten()
        # 定义全连接层
        # self.fc = torch.nn.Linear(128, 256)

    def forward(self, inputs):
        # 将输入的形状转换为(batch_size,channels,length)
        inputs = inputs.view(inputs.shape[0], -1, 1)
        # 运行CNN模型
        out = self.cnn(inputs)
        # 将CNN模型的输出展平
        out = self.flatten(out)
        # 将展平后的输出运行全连接层
        # out = self.fc(out)
        # 返回输出
        return out


class Bi_LSTM(nn.Module):
    # 初始化函数
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        # 定义LSTM层，输入维度为256，隐藏维度为128，层数为2，batch_first为True，dropout为0.4，双向为True
        self.lstm_1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        # 定义Dropout层，dropout比例为0.4
        self.dropout = nn.Dropout(0.4)
        # 定义线性层，输入维度为256，输出维度为128
        self.linear_1 = nn.Linear(256, 128)
        # 定义激活函数，为tanh
        self.act = nn.Tanh()
        # 定义Flatten层
        self.flatten = nn.Flatten()

    # 定义前向传播函数
    def forward(self, inputs):
        # 将输入的维度转换为（batch_size,1,seq_len）
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        # 运行LSTM层
        out, _ = self.lstm_1(inputs)
        # 运行Dropout层
        out = self.dropout(out)
        # 取LSTM层的最后一行
        out = out[:, -1, :]
        # 运行线性层
        out = self.linear_1(out)
        # 返回结果
        return out


class GMOVE(nn.Module):
    # 初始化函数
    def __init__(self):
        super(GMOVE, self).__init__()
        # 定义Bi-LSTM层
        self.Bi_LSTM = Bi_LSTM()
        self.CNN_GE = CNN_GE()
        self.CNN_METER = CNN_METER()
        # 实例化Dropout
        self.dropout = torch.nn.Dropout(0.4)
        # 实例化ReLU
        self.act = torch.nn.ReLU()
        # 实例化Flatten
        self.flatten = nn.Flatten()
        # 实例化Linear
        self.fc1 = torch.nn.Linear(384, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x1, x2, x3, label):
        # 使用Bi_LSTM对x1进行编码
        encoder_out1 = self.Bi_LSTM(x1)
        # 使用CNN对x2、x3编码
        encoder_out2 = self.CNN_GE(x2)
        encoder_out3 = self.CNN_METER(x3)

        out = torch.cat([encoder_out1, encoder_out2, encoder_out3], dim=1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        # out = torch.tanh(out)
        # out = self.dropout(out)
        # out = self.fc4(out)
        # out = self.dropout(out)
        # out = torch.tanh(out)
        # out = self.fc5(out)
        # out = self.dropout(out)
        out = torch.sigmoid(out)
        out = out.squeeze()
        return out

