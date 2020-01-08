import sys
sys.path.append('./')

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np 
import pandas as pd
import time

from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score

from model import rnn_turning_point

from utils.opfiles import *
from utils.split_data import split
from utils.dataSet import Indicator


def train_rnn_turning_point(window=20, window_list=[6], machineID = '207776314',isRaw=False, isCurr=False, modelName="FEMT_LSTM"):

    TP = Indicator(window, window_list, machineID, isRaw, isCurr)
    splited_data = TP.splited_data

    #print("train/validation/test split: {}/{}/{}".format(splited_data["train_data"].shape, splited_data["validation_data"].shape, splited_data["test_data"].shape))
    train_data, train_labels = splited_data["train_data"], splited_data["train_labels"]
    val_data, val_labels = splited_data["validation_data"], splited_data["validation_labels"]
    test_data, test_labels = splited_data["test_data"], splited_data["test_labels"]

    datas = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    clf = rnn_turning_point.build_train(datas, machineID, modelName)
    
    # pred test
    pred = clf.predict(test_data).reshape(-1, )
    pred = (pred > 0.5).astype(int)
    test_labels = test_labels.reshape(-1, )
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average="binary")
    p = precision_score(test_labels, pred)
    r = recall_score(test_labels, pred)
    print("test ---- f1, acc: ", f1, acc)
    print(classification_report(test_labels, pred))
    print("f1:", f1)
    print("p:", p)
    print("r:", r)
    return clf, acc, f1, p, r


def test_rnn_turning_point(window=20, window_list=[6], machineID = '207776314', isRaw=False, isCurr=False, modelName="FEMT_LSTM"):

    TP = Indicator(window, window_list, machineID, isRaw, isCurr)
    splited_data = TP.splited_data

    #print("train/validation/test split: {}/{}/{}".format(splited_data["train_data"].shape, splited_data["validation_data"].shape, splited_data["test_data"].shape))
    train_data, train_labels = splited_data["train_data"], splited_data["train_labels"]
    val_data, val_labels = splited_data["validation_data"], splited_data["validation_labels"]
    test_data, test_labels = splited_data["test_data"], splited_data["test_labels"]

    datas = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    clf = rnn_turning_point.build_test(datas, machineID, modelName=modelName)

    # pred test
    pred = clf.predict(test_data).reshape(-1, )
    pred = (pred > 0.5).astype(int)
    test_labels = test_labels.reshape(-1, )
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average="binary")
    p = precision_score(test_labels, pred)
    r = recall_score(test_labels, pred)
    print("test ---- f1, acc: ", f1, acc)
    print(classification_report(test_labels, pred))
    print("f1:", f1)
    print("p:", p)
    print("r:", r)
    np.save('./lib_baseline/lib_{}/model_pred_rnn_{}.npy'.format(modelName, machineID), pred)
    return clf, acc, f1, p, r


if __name__ == "__main__":
     import sys
     dataSetID = sys.argv[1]
     modelName = sys.argv[2]

     # check model name
     if modelName == 'FEMT_LSTM':
          isRaw = False
     elif modelName == 'S_LSTM':
          isRaw = False
     elif modelName == 'P_LSTM':
          isRaw = True
     else:
          print("modelName must be:")
          print("FEMT_LSTM or S_LSTM or P_LSTM")
          assert 1 == 0, "modelName error!"

     # check machine ID
     if dataSetID == 'M1':
          machineID = '207776314'
     elif dataSetID == 'M2':
          machineID = '908054'
     elif dataSetID == 'M3':
          machineID = '1273805'
     else:
          print("dataSetID must be:")
          print("M1 or M2 or M3")
          assert 1 == 0, "dataSetID error!"
     
     # set window parameter
     if modelName == 'FEMT_LSTM':
          if dataSetID == 'M3':
               window = 5
          else:
               window = 10
     if modelName == 'S_LSTM':
          if dataSetID == 'M1':
               window = 5
          else:
               window = 10
     if modelName == 'P_LSTM':
          window = 10
     
     test_rnn_turning_point(window=window, window_list=[3], machineID=machineID, isRaw=isRaw, isCurr=True, modelName=modelName)