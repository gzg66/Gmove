# Recommending Move Method Refactoring Opportunities Based on Feature Fusion and Deep Learning

This repository contains the source code and datasets required to reproduce the experimental results presented in the paper **"Recommending Move Method Refactoring Opportunities Based on Feature Fusion and Deep Learning"**.

## Paper Information

* **Title**: Recommending Move Method Refactoring Opportunities Based on Feature Fusion and Deep Learning
* **Authors**: Yang Zhang¹, Zhenggang Gu¹, Nan Zhang², Kun Zheng¹
* **Affiliations**:
    1. School of Information Science and Engineering, Hebei University of Science and Technology, Shijiazhuang, Hebei, China
    2. HBIS Digital Technology Co. Ltd., Shijiazhuang, Hebei, China
* **ID**: 10060

## Repository Structure

### Source Code

* **Automation Script**:
* `RQ.bat`: An automated experiment script that covers **RQ1**, **RQ2**, **RQ4**, and **RQ5**.
* **Note for RQ3**: To reproduce the ablation study (**RQ3**), users need to manually modify the `forward` method in `model.py` to concat specific feature vectors (e.g., only `encoder_out1` and `encoder_out2` for the CE+GE variant).


* **Deep Learning Model (GMove)**:
  * `model.py`: Defines the `GMove` architecture, including Bi-LSTM (semantic), CNN_GE (structural), and CNN_METER (metric) branches.
  * `train.py`: Contains the training loop and validation logic.
  * `data.py`: Handles data loading and preprocessing.
  * `main.py`: The entry point for training the GMove model on the synthetic dataset (corresponding to **RQ2**).
  * `test.py`: The evaluation script for real-world projects (corresponding to **RQ5**).
* **Baseline Models (Machine Learning)**:
* `LogisticRegression.py`, `DecisionTree.py`, `RandomForest.py`, `ExtraTrees.py`, `SVC.py`, `NB.py` (Naive Bayes), `xgb.py` (XGBoost): Scripts to train and evaluate baseline models used in **RQ1** and **RQ4**.

### Datasets (Required)

To reproduce the results, please ensure the data is placed in the `data/` directory:

* `data/train_data/`: Should contain the generated dataset csv files for training.
* `data/test_data/all_test_data.csv`: Contains the consolidated test dataset for real-world projects used in **RQ5**.
