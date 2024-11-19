#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load_util import get_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import sys
import numpy as np
import lightgbm as lgb


def get_lgb(x_train, x_test, y_train, y_test, num_round=100):
    param = {
        'num_leaves': 180,
        'objective': 'multiclass',
        'learning_rate': 0.05,
        'num_iterations': 1000,
        'num_class': 5,
        'num_threads': 2,
        'max_depth': 18,
        'scale_pos_weight': 1,
        'random_state': 1332,
        'subsample': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'early_stopping_round': 20
    }
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_test, label=y_test)
    param['metric'] = 'multi_error'
    model = lgb.train(param, train_data, num_round, valid_sets=[valid_data])
    return model


def print_metrics(y_test, y_pred, stage="test set"):
    # # calculate accuracy\recall\precision\f1-score
    y_pred = np.argmax(y_pred, axis=1)
    print("y_test:", y_test)
    print("-" * 50)
    print("-" * 50)
    print("current stage:"+stage)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:%6.2f%%' % (accuracy * 100))
    recall = recall_score(y_test, y_pred, average='micro')
    print('recall:%6.2f%%' % (recall * 100))
    precision = precision_score(y_test, y_pred, average='micro')
    print('precision:%6.2f%%' % (precision * 100))
    f1 = f1_score(y_test, y_pred, average='micro')
    print('f1:%6.2f%%' % (f1 * 100))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = np.sum(cm, axis=1)
    cm_pct = cm / cm_sum[:, None]
    print("confusion matrix:")
    cm_pct = str(cm_pct).replace("[", "").replace("]", "")
    cm_pct = " " + cm_pct
    print(cm_pct)
    print("confusion matrix:")
    cm_str = str(cm).replace("[", "").replace("]", "")
    print(cm_str)


def main():
    # load data
    if len(sys.argv) > 1:
        train_data_path = sys.argv[1]
        model_path = sys.argv[2]
    print(train_data_path)
    print(model_path)

    # get data
    X, y = get_data(train_data_path)
    print("x length:" + str(X.shape))
    print("y length:" + str(y.shape))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=12343)

    # train model
    model = get_lgb(X_train, X_test, y_train, y_test)
    model.save_model(model_path)

    # predict test data
    y_train_pred = model.predict(X_train)
    print_metrics(y_train, y_train_pred, stage="train set")
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred)


if __name__ == '__main__':
    main()

