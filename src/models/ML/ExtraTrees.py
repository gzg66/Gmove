#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   ExtraTrees.py
"""
import sys
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
# 假设你的数据集已经准备好，特征为X，标签为y
# 进行训练集和测试集的划分
file_path = sys.argv[1]
data = pd.read_csv(file_path, header=None)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# 创建SVC模型
model = ExtraTreesClassifier()

# 对训练集训练模型
model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出准确率、召回率、F1值
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
