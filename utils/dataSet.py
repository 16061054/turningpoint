import sys
sys.path.append('../')

import numpy as np 
import pandas as pd
import time

from utils.opfiles import *
from utils.split_data import split

import tsfresh as tsf
import talib


class Indicator(object):

    def __init__(self, window, window_list=[6], machineID = '207776314', isRaw=False, isCurr=False):
        """
        window: rnn的截断长度
        window_list: 计算特征时的窗口长度
        isRaw: 是否只使用原始cpu数据
        """
        self.window_list = window_list
        self.machineID = machineID
        self.isRaw = isRaw
        self.isCurr = isCurr

        self.define_dir()
        self.define_machine(machineID)
        self.define_label()
        if not isCurr:
            self.get_feature()
        else:
            self.get_feature_cur()
        
        if not isRaw:
            self.prepare_data_metrics_matrix(window)
        else:
            self.prepare_data_raw(window)


    def define_dir(self):
        # define dirs
        self.data_dir = './data_org'
        self.base_dir = '.'
        self.cache_dir = os.path.join(self.base_dir, 'cache')
        if not os.path.exists(self.cache_dir) :
            raise Exception(u"需要先产生分段数据!")


    def define_machine(self, machineID = '207776314', TD = 1, max_error = 0.0015):
        """
        machineID：207776314 908054 451283682 1273805
        """
        machineID_cache_dir = os.path.join(self.cache_dir, machineID) # 为每一个machineID单独创建cache dir
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
            # print("change to label 2:")
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

        
    def get_feature(self):
        data = self.feature[:,0]
        labels = self.labels

        data_df = pd.DataFrame()
        data_df['cpu'] = data[:-1] # 错开一步
        data_df['y'] = labels[1:] # 错开一步
        self.data_df = data_df

        for w in self.window_list:
            # group1 绝对特征
            self.moving_average(data_df, n=w)
            self.var_value(data_df, n=w)
            # group2 波动特征
            self.rsi(data_df, n=w)
            self.absolute_sum_of_changes(data_df, n=w)
            self.mean_second_derivative_central(data_df, n=w)
            self.linear_trend(data_df, n=w)
        
        n = self.window_list[0]
        self.data_df = self.data_df[['cpu', 'y','rolling_mean_{}'.format(n), 'var_value_{}'.format(n), 'RSI_{}'.format(n), 'asc_{}'.format(n), 'msdc_{}'.format(n), 'lt_{}'.format(n)]]
        self.data_df = self.data_df.dropna()

    
    def get_feature_cur(self):
        data = self.feature[:,0]
        labels = self.labels

        data_df = pd.DataFrame()
        data_df['cpu'] = data[:] # 不错开一步
        data_df['y'] = labels[:] # 不错开一步
        self.data_df = data_df

        for w in self.window_list:
            # group1 绝对特征
            self.moving_average(data_df, n=w)
            self.var_value(data_df, n=w)
            # group2 波动特征
            self.rsi(data_df, n=w)
            self.absolute_sum_of_changes(data_df, n=w)
            self.mean_second_derivative_central(data_df, n=w)
            self.linear_trend(data_df, n=w)
        
        n = self.window_list[0]
        self.data_df = self.data_df[['cpu', 'y','rolling_mean_{}'.format(n), 'var_value_{}'.format(n), 'RSI_{}'.format(n), 'asc_{}'.format(n), 'msdc_{}'.format(n), 'lt_{}'.format(n)]]
        self.data_df = self.data_df.dropna()



    def prepare_data_raw(self, window):
        """
        """
        X = self.data_df['cpu'].values.reshape(-1,)
        y = self.data_df['y'].values.reshape(-1,)

        # normalize data
        feature_max = np.max(X, axis=0)
        feature_min = np.min(X, axis=0)
        X = (X - feature_min)/(feature_max - feature_min)
        normalize_map = {"max":feature_max, "min":feature_min}

        # rnn根据窗口长度划分数据 已经确保了X和y的对应关系
        continous_X = []
        continous_y = []
        for index, data in enumerate(X):
            if index >= window:
                tmp = X[index - window: index]
                continous_X.append(tmp)
                continous_y.append(y[index-1])

        continous_X = np.array(continous_X).reshape(-1, window, 1)
        continous_y = np.array(continous_y).reshape(-1, 1)
        
        self.X = continous_X # shape (-1, window, 1)
        self.y = continous_y # shape (-1, 1)
        self.normalize_map = normalize_map
        self.splited_data = split(self.X, self.y)


    def prepare_data_metrics_matrix(self, window):
        """
        注意固定顺序

        如果需要测试之前训练好的lib_joint_* 就打开注释， 替换固定列的情况
        """
        n = self.window_list[0]
        # X =  self.data_df[self.data_df.columns.difference(['y'])].values
        X = self.data_df[['cpu','rolling_mean_{}'.format(n), 'var_value_{}'.format(n), 'RSI_{}'.format(n), 'asc_{}'.format(n), 'msdc_{}'.format(n), 'lt_{}'.format(n)]].values
        y = self.data_df['y'].values

        # normalize data
        feature_max = np.max(X, axis=0)
        feature_min = np.min(X, axis=0)
        X = (X - feature_min)/(feature_max - feature_min)
        normalize_map = {"max":feature_max, "min":feature_min}

        # rnn根据窗口长度划分数据 已经确保了X和y的对应关系
        continous_X = []
        continous_y = []
        for index, data in enumerate(X):
            if index >= window:
                tmp = X[index - window: index]
                continous_X.append(tmp)
                continous_y.append(y[index-1])
                
        continous_X = np.array(continous_X) 
        continous_y = np.array(continous_y).reshape(-1, 1) 
        
        self.X = continous_X # shape (-1, window, feature)
        self.y = continous_y # shape (-1, 1)
        self.normalize_map = normalize_map
        self.splited_data = split(self.X, self.y)



    # group1 滑动平均特征
    def moving_average(self, data_df, n=12):
        data_df['rolling_mean_{}'.format(n)] = talib.SMA(data_df['cpu'], timeperiod=n)

    # group2 绝对值特征
    def var_value(self, data_df, n=12):
        data_df['var_value_{}'.format(n)] = data_df['cpu'].rolling(n).apply(lambda x: np.std(x))

    # group4 波动性特征
    def absolute_sum_of_changes(self, data_df, n=12):
        data_df['asc_{}'.format(n)] = data_df['cpu'].rolling(n).apply(lambda x: tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(x))

    def mean_second_derivative_central(self, data_df, n=12):
        data_df['msdc_{}'.format(n)] = data_df['cpu'].rolling(n).apply(lambda x: tsf.feature_extraction.feature_calculators.mean_second_derivative_central(x))

    def linear_trend(self, data_df, n=12):
        def _get(x):
            tmp = tsf.feature_extraction.feature_calculators.linear_trend(x, param=[{'attr':'slope'}])
            return tmp[0][1]

        data_df['lt_{}'.format(n)] = data_df['cpu'].rolling(n).apply(lambda x: _get(x))

    def rsi(self, data_df, n=12):
        data_df['RSI_{}'.format(n)] = talib.RSI(data_df['cpu'], timeperiod=n)