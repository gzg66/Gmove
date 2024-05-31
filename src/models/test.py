#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   test.py
"""
import sys
import os
from datetime import time

import numpy as np
import pandas as pd
import torch

import model
import data
import train

if __name__ == '__main__':
    model_name_method = "GMOVE"
    real_path = sys.argv[1]
    real = ['weka', 'ant', 'freecol', 'jmeter', 'freemind', 'jtopen', 'drjava', 'maven']
    for i in real:
        path_method = real_path + i + '.csv'
        config_ = model.config(model_name_method)

        test_iter = data.get_dataLoad_test(config_, path_method)
        print('测试集大小-----》', len(test_iter.dataset))

        model_method = model.GMOVE().to(config_.device)
        print('*' * 50 + i + '*' * 50)
        train.test(config_, model_method, test_iter)
        print('*' * 120)
