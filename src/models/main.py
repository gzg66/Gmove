#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   main.py
"""
import sys
import torch
import numpy as np
import time
import data
import train
import warnings
import sys

import model


# 定义一个Logger类，用于输出日志
class Logger(object):
    # 初始化Logger类，filename为日志文件名，stream为输出流，默认为sys.stdout
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    # 写入日志
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # 刷新日志
    def flush(self):
        return True


warnings.filterwarnings("ignore")
if __name__ == '__main__':
    model_name_method = "GMOVE3"
    path_method = sys.argv[1]
    sys.stdout = Logger(
        filename='./' + model_name_method + '_3layer_30epoch_256_BNLayer_1.txt',
        stream=sys.stdout)

    config_ = model.config(model_name_method)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    print('数据集加载')
    start_time = time.time()
    train_iter, test_iter = data.get_dataLoad(config_, path_method)
    time_dif = data.get_time_dif(start_time)
    print('训练集大小-----》', len(train_iter.dataset))
    print('测试集大小-----》', len(test_iter.dataset))
    print("模型开始之前，准备数据总时间：", time_dif)

    start_train = time.time()
    print('###############################################################')
    model_method = model.GMOVE().to(config_.device)
    print('model_method:', model_method)
    train.train(config_, model_method, train_iter)
    train.test(config_, model_method, test_iter)