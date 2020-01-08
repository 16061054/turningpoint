import numpy as np 
import pandas as pd
import os
import pickle

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve,precision_recall_fscore_support, precision_score, recall_score
import time

from utils.opfiles import load_pickle, write_pickle, write_txt
from utils.split_data import split
import talib

import warnings
warnings.filterwarnings("ignore")

class Indicator_SVM(object):

    def __init__(self, window_list=[6], machineID='207776314'):
        self.window_list = window_list
        self.define_dir()
        self.define_machine(machineID)
        self.define_label()
        self.get_3_basic_feature()
        self.prepare_data_ml()


    def define_dir(self):
        # define dirs
        self.data_dir = './data_org'
        self.base_dir = '.'
        self.cache_dir = os.path.join(self.base_dir, 'cache')
        if not os.path.exists(self.cache_dir) :
            raise Exception(u"需要先建立缓存目录：%s!" % self.cache_dir)


    def define_machine(self, machineID = '207776314', TD = 1, max_error = 0.0015):
        """
        # 207776314 908054 1273805
        """
        # 为每一个machineID单独创建cache dir
        machineID_cache_dir = os.path.join(self.cache_dir, machineID)
        cached_seg_file_name_prefix = machineID + 'TD_ ' + str(TD) +'_' + str(max_error)
        target_file_path = os.path.join(machineID_cache_dir, cached_seg_file_name_prefix + '_segments_list.pkl')
        if not os.path.exists(target_file_path) :
            raise Exception(u"需要先产生分段数据!")
        # res = {'segments': segments, 'data':data}
        res = load_pickle(target_file_path)

        self.segments_list = res["segments"]
        self.data = res["data"]
        self.feature = np.load("./data_org/features_7_{}.npy".format(machineID))
        assert self.feature.shape[1] == 7

    
    def define_label(self, label2=True):
        data = self.data
        segments_list = self.segments_list

        labels = np.zeros((len(data),))

        for idx, seg in enumerate(segments_list[1:]):
            x1,y1,x2,y2 = seg
            front = segments_list[idx]
            x0,y0,x11,y11 = front
            
            if(y0 < y1 and y1 > y2):
                labels[x1] = 2 # peak

            elif(y0 > y1 and y1 < y2):
                labels[x1] = 1 # valley

        count_p = np.sum(np.array(labels == 2).astype(int))
        count_v = np.sum(np.array(labels == 1).astype(int))
        count_s = np.sum(np.array(labels == 0).astype(int))
        total = len(labels)
        # print("count_p", count_p, count_p/total)
        # print("count_v", count_v, count_v/total)
        # print("count_s", count_s, count_s/total)
        self.labels = labels
        self.count_p = count_p
        self.count_v = count_v
        self.count_s = count_s

        if label2:
            # print("---------------change to label 2---------------")
            labels_2 = np.zeros((len(data),))
            for i, d in enumerate(labels):
                if d == 2 or d == 1:
                    labels_2[i] = 1
        
            count_t = np.sum(np.array(labels_2 == 1).astype(int))
            count_c = np.sum(np.array(labels_2 == 0).astype(int))
            total = len(labels)
            # print("count_t", count_t, count_t/total)
            # print("count_s", count_c, count_c/total)
            self.labels = labels_2 # 覆盖掉原来的label
            self.count_t = count_t
            self.count_s = count_s


    def get_3_basic_feature(self):
        data = self.feature[:,0]
        labels = self.labels

        data_df = pd.DataFrame()
        data_df['cpu'] = data[:] # 不错开一步
        data_df['y'] = labels[:] # 不错开一步
        self.data_df = data_df
        
        for w in self.window_list:
            # group1 滑动平均特征
            self.moving_average(data_df, n=w)
            # group2 绝对值特征
            self.var_value(data_df, n=w) # 6 9 12

        self.data_df = self.data_df.dropna()


    def prepare_data_ml(self):
        """prepare data for continous model.
        直接返回每个点对应的几个特征
        """
        
        continous_X = self.data_df[self.data_df.columns.difference(['y'])].values
        y = self.data_df['y'].values
        y = np.array(y).reshape(-1, 1) # shape (-1,1)
        
        # normalize 
        feature_max = np.max(continous_X, axis=0)
        feature_min = np.min(continous_X, axis=0)
        continous_X = (continous_X - feature_min)/(feature_max - feature_min)
        normalize_map = {"max":feature_max, "min":feature_min}

        self.X = continous_X
        self.y = y
        self.normalize_map = normalize_map
        self.splited_data = split(self.X, self.y)


    # group1 滑动平均特征
    def moving_average(self, data_df, n=12):
        """
        n是滑动平均的个数， 默认一小时, 分短期和长期
        """
        data_df['rolling_mean_{}'.format(n)] = talib.SMA(data_df['cpu'], timeperiod=n)


    def var_value(self, data_df, n=12):
        data_df['var_value_{}'.format(n)] = data_df['cpu'].rolling(n).apply(lambda x: np.std(x))


class BaseLine_SVM(object):
    def __init__(self, window_list=[6], machineID='207776314'):
        self.ic = Indicator_SVM(window_list, machineID)
        self.splited_data = self.ic.splited_data
        self.model_ckpt_path = 'lib_baseline/lib_SVM_Basic/svm_basic_%s' % machineID
        self.model_info_path = 'lib_baseline/lib_SVM_Basic/svm_basic_info_%s.txt' % machineID
        

    def train_main(self):
        C_range = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10, 15, 20]
        gamma_range = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10, 15, 20]

        cnt = 0
        total = len(C_range) * len(gamma_range)
        self.best_f1 = 0
        self.record_param = []
        self.best_index = -1
        self.best_clf = None

        for C in C_range:
            for gamma in gamma_range:
                
                clf, acc, f1, p, r = self.train_a_model(splited_data=self.splited_data, C=C, gamma=gamma, is_test=False)
                self.record_param.append([C, gamma, acc, f1, p, r])
                print("No.{}/{} --- acc/f1/ {:.4}/{:.4}/ --- C/gamma {}/{} ".format(cnt, total, acc, f1, C, gamma))
                if  f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_index = cnt
                    self.best_clf = clf
                cnt += 1

        C = self.record_param[self.best_index][0]
        gamma = self.record_param[self.best_index][1]
        clf, acc, f1, p, r = self.train_a_model(self.splited_data, C, gamma, is_test=True)
        self.clf = clf
        self.acc = acc
        self.f1 = f1
        self.p = p
        self.r = r
        # save model
        write_pickle(clf, self.model_ckpt_path)


    def train_a_model(self, splited_data, C, gamma, is_test=False):

        train_data, train_labels = splited_data["train_data"], splited_data["train_labels"]
        val_data, val_labels = splited_data["validation_data"], splited_data["validation_labels"]
        test_data, test_labels = splited_data["test_data"], splited_data["test_labels"]

        clf = SVC(gamma=gamma, C=C, class_weight='balanced') # 'gamma='auto', C=1.0
        clf.fit(train_data, train_labels)
        
        if not is_test:
            # pred valid
            pred = clf.predict(val_data).reshape(-1, 1)
            acc = accuracy_score(val_labels, pred)
            p = precision_score(val_labels, pred)
            r = recall_score(val_labels, pred)
            f1 = f1_score(val_labels, pred, average='binary')
            return clf, acc, f1, p, r
        else:
            # pred test
            pred = clf.predict(test_data).reshape(-1, 1)
            acc = accuracy_score(test_labels, pred)
            p = precision_score(test_labels, pred)
            r = recall_score(test_labels, pred)
            f1 = f1_score(test_labels, pred, average='binary')
            self.rep = classification_report(test_labels, pred)
            return clf, acc, f1, p, r


    def test_a_model(self):
        clf = load_pickle(self.model_ckpt_path)
        splited_data = self.splited_data
        test_data, test_labels = splited_data["test_data"], splited_data["test_labels"]

        pred = clf.predict(test_data).reshape(-1, 1)
        acc = accuracy_score(test_labels, pred)
        p = precision_score(test_labels, pred)
        r = recall_score(test_labels, pred)
        f1 = f1_score(test_labels, pred, average='binary')
        print("f1:{:.4} p:{:.4} r:{:.4} acc:{:.4}".format(f1, p, r, acc))
        write_txt("f1:{:.4} p:{:.4} r:{:.4} acc:{:.4}".format(f1, p, r, acc), self.model_info_path)


def main(window_list=[3], machineID='207776314', isTrain=False):
    bsvm = BaseLine_SVM(window_list=window_list, machineID=machineID)
    if isTrain:
        # train
        bsvm.train_main()
        print("valid set info: ")
        print(bsvm.best_index)
        print(bsvm.record_param[bsvm.best_index])
        print("test set info:")
        print(bsvm.rep)
        print("self.f1", bsvm.f1)
        print("self.acc", bsvm.acc)
        print("self.p", bsvm.p)
        print("self.r", bsvm.r)
        # test
        print("load and test:")
        bsvm.test_a_model()
    else:
        # test
        print("load and test:")
        bsvm.test_a_model()


if __name__ == '__main__':
    main(window_list=[3], machineID='207776314', isTrain=False)
    main(window_list=[3], machineID='908054', isTrain=False)
    main(window_list=[3], machineID='1273805', isTrain=False)