#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File       :   train.py
"""
import sys

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, roc_curve, \
    precision_recall_curve, jaccard_score, roc_auc_score

# from models.model import XGBoostModel


# 定义一个Logger类，用于记录输出信息
class Logger(object):
    # 初始化函数，filename为文件名，stream为输出流，默认为sys.stdout
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    # 写入信息到输出流和日志文件
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # 刷新输出流
    def flush(self):
        return True


def train(config, model, train_iter):
    # 设置模型为训练模式
    model.train()
    # 获取模型的参数
    param_optimizer = list(model.named_parameters())
    # 定义不进行衰减的参数
    no_decay = ['bias', 'LayerNorm', 'LayerNorm.weight']
    # 获取需要进行衰减的参数
    optimizer_growped_paramters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 初始化优化器
    optimizer = optim.Adam(params=optimizer_growped_paramters,
                           lr=config.learning_rate)
    # 记录是否很久没有提升效果
    flag = False  # 记录是否很久没有提升效果

    # 开始训练
    model.train()
    # 初始化预测结果和标签结果
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 初始化训练准确率、损失、FPR、TPR
    train_acc_list = np.array([], dtype=int)
    train_loss_list = np.array([], dtype=int)
    fpr_list = np.array([], dtype=int)
    tpr_list = np.array([], dtype=int)

    # 开始训练
    for epoch in range(config.num_epochs):
        print('---------------------Epoch{}/{}--------------------'.format(epoch + 1, config.num_epochs))
        epoch_train_acc_list_1 = np.array([], dtype=int)
        epoch_train_loss_list = np.array([], dtype=int)
        epoch_fpr_list = np.array([], dtype=int)
        epoch_tpr_list = np.array([], dtype=int)

        # 遍历训练数据
        for i, (x1, x2, x3, labels) in enumerate(train_iter):
        # for i, (x1, x2, labels) in enumerate(train_iter):
            # 将训练数据转换到设备
            x1 = x1.to(config.device)
            x2 = x2.to(config.device)
            x3 = x3.to(config.device)
            labels = labels.to(config.device)
            # 计算模型输出
            outputs = model(x1, x2, x3, labels)
            # outputs = model(x1, x2, labels)
            # 梯度归零
            model.zero_grad()
            # 计算损失
            loss = F.binary_cross_entropy(outputs, labels)
            # 反向传播
            loss.backward()
            # 计算损失
            loss = loss.detach().cpu().numpy()
            # 将损失添加到训练损失列表
            epoch_train_loss_list = np.append(epoch_train_loss_list, loss)
            # 更新参数
            optimizer.step()
            # 将输出转换为标签
            predict = (outputs >= 0.5).long()  # 将输出阈值设置为0.5 [0.49,0.58]
            # 将标签转换为numpy
            labels = labels.data.cpu().numpy()
            # 将预测结果转换为numpy
            predict = predict.cpu().numpy()

            # 判断是否是第一次训练
            if i != 0:
                # 获取预测结果和标签结果的长度
                j = len(predict_all)
                # 将标签结果添加到labels_all
                labels_all = np.insert(labels_all, j, labels, axis=0)
                # 将预测结果添加到predict_all
                predict_all = np.insert(predict_all, j, predict, axis=0)
            else:
                # 获取预测结果和标签结果的长度
                j = len(predict)
                # 将标签结果赋值给labels_all
                labels_all = labels
                # 将预测结果赋值给predict_all
                predict_all = predict
            # 计算准确率
            acc = metrics.accuracy_score(labels_all[:, ], predict_all[:, ])
            # 将准确率添加到训练准确率列表
            epoch_train_acc_list_1 = np.append(epoch_train_acc_list_1, acc)

        # 打印训练结果
        print("accuracy macro:{} \t loss:{} \t ".format(np.mean(epoch_train_acc_list_1), loss.item()))
        # 将训练准确率添加到训练准确率列表
        train_acc_list = np.append(train_acc_list, np.mean(epoch_train_acc_list_1))
        # 将训练损失添加到训练损失列表
        train_loss_list = np.append(train_loss_list, np.mean(epoch_train_loss_list))
        # 将FPR添加到FPR列表
        fpr_list = np.append(fpr_list, np.mean(epoch_fpr_list))
        # 将TPR添加到TPR列表
        tpr_list = np.append(tpr_list, np.mean(epoch_tpr_list))

        # nums = evaluate(config, model, val_iter, False)
        # print('----------------------开始验证----------------------')
        # print('精确率--->', nums[-3])
        # print('召回率--->', nums[-2])
        # print('F1分数--->', nums[-1])
        # model.train()
        # 判断是否很久没有提升效果
        if flag:
            break
    # 保存模型
    torch.save(model.state_dict(), config.save_path)
    # 返回训练准确率、损失、FPR、TPR
    return train_acc_list, train_loss_list, fpr_list, tpr_list


# 定义评估函数，用于评估模型在验证集上的表现
def evaluate(config, model, vaild_iter, test=False):
    # 将模型设置为评估模式
    model.eval()
    # 初始化开发集上的损失、预测结果和标签
    dev_loss_all = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 初始化开发集上的损失
    dev_loss = np.array([], dtype=int)
    # 不计算梯度
    with torch.no_grad():
        # 遍历验证集
        for i, (x1, x2, x3, labels) in enumerate(vaild_iter):
        # for i, (x1, x2, labels) in enumerate(vaild_iter):
            # 将输入数据转换到设备
            x1 = x1.to(config.device)
            x2 = x2.to(config.device)
            x3 = x3.to(config.device)
            labels = labels.to(config.device)
            # 计算模型输出
            outputs = model(x1, x2, x3, labels)
            outputs = outputs.view([1])
            # outputs = model(x1, x2, labels)
            # 计算预测结果
            predict = (outputs >= 0.5).long()
            # 计算开发集上的损失
            epoch_dev_loss = F.binary_cross_entropy(outputs, labels)
            epoch_dev_loss = epoch_dev_loss.detach().cpu().numpy()
            # 将标签、预测结果和损失从设备转换到CPU
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            # 如果不是第一次遍历，则将标签、预测结果和损失添加到对应变量中
            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all, j, labels, axis=0)
                predict_all = np.insert(predict_all, j, predict, axis=0)
            else:
                labels_all = labels
                predict_all = predict
            dev_loss_all = np.append(dev_loss_all, epoch_dev_loss)

    # 如果test参数为True，则计算模型在验证集上的指标
    if test:
        # 计算模型在验证集上的指标
        subsetAccuracy, macroPrecision, macroRecall, macroF1, macroAuc = metric(labels_all, predict_all)

        confusion =metrics.confusion_matrix(labels_all, predict_all)
        # 返回模型在验证集上的指标
        return subsetAccuracy, macroPrecision, macroRecall, macroF1, macroAuc, confusion

    # 计算模型在验证集上的准确率
    acc1 = metrics.accuracy_score(labels_all[:, ], predict_all[:, ])
    # 将开发集上的损失添加到开发集上的损失变量中
    dev_loss = np.append(dev_loss, np.mean(dev_loss_all))

    # 计算精确率
    precision = precision_score(labels_all, predict_all)
    # 计算召回率
    recall = recall_score(labels_all, predict_all)
    # 计算F1分数
    f1 = f1_score(labels_all, predict_all)
    # 返回模型在验证集上的指标
    return acc1, dev_loss, hamming_loss, precision, recall, f1


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    subsetAccuracy, macroPrecision, macroRecall, macroF1, macroAuc, confusion = evaluate(config, model, test_iter, True)
    print("#" * 20, '测试结果', "#" * 20)
    print("准确率:", subsetAccuracy)
    print("精确率:", macroPrecision)
    print("召回率:", macroRecall)
    print("F1:", macroF1)
    print("AUC:", macroAuc)
    print("混淆矩阵:")
    print(confusion)


def metric(y_true, y_pre, sample_weight=None):
    def SubAccuracy():
        subsetAccuracy = accuracy_score(y_true, y_pre)
        return subsetAccuracy

    def MacPrecision():
        macroPrecision = precision_score(y_true, y_pre)
        return macroPrecision

    def MacRecall():
        macroRecall = recall_score(y_true, y_pre)
        return macroRecall

    def MacF1():
        macroF1 = f1_score(y_true, y_pre)
        return macroF1

    def macroAUC():
        macroAuc = roc_auc_score(y_true, y_pre)
        return macroAuc

    return SubAccuracy(), MacPrecision(), MacRecall(), MacF1(), macroAUC()