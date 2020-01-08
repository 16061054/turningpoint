import numpy as np
import tsfresh as tsf


def exponential_smoothing(s, N):
    '''
    一次指数平滑
    :param N:  窗长
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    
    alpha = 2.0/(N+1)
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp


def weighted_mean(s):
    """
    加权平均
    """
    N = len(s)
    weight  = list(range(N, 0, -1))
    ans = [i*j for (i, j) in zip(s, weight)]
    ans = sum(ans)
    ans = ans*(2.0/((1+N)*N))
    return ans


def absolute_sum_of_changes(s):
    return tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(s)


def mean_abs_change(s):
    return tsf.feature_extraction.feature_calculators.mean_abs_change(s)


def mean_second_derivative_central(s):
    """
    Returns the mean value of a central approximation of the second derivative
    """
    return tsf.feature_extraction.feature_calculators.mean_second_derivative_central(s)


def extract_feature_window(seq, window):
    """
    基于窗口提取特征， 只提取当前点对应的窗口里边的特征
    
    # group 1 绝对值特征
    1.  窗口中的最小值
    2.  窗口中的最大值
    3.  当前值
    4.  窗口内的均值
    5.  窗口内的方差
    6.  Exponential Moving Average list
    7.  weighted mean
    8.  count_over_mean

    # group 2 相对值特征
    9.  The index for the type of K-line (ITL) 窗口中的 第一个值 < 最后一个值 ？ 1：-1
    10. The change rate of transaction money compared to the previous trading day (当前值 - 最近值)/最近值
    11. 当前值在窗口中的相对位置 ((pc (t) − pl (t)) − (ph (t) − pc (t))) / (ph (t) − pl (t))

    # group 3 tsf
    12. absolute_sum_of_changes
    13. mean_abs_change

    15. mean_second_derivative_central
    """
    # group 1 绝对值特征
    max_value = np.max(seq)
    min_value = np.min(seq)
    cur_value = seq[-1]
    mean_value = np.mean(seq)
    var_value = np.var(seq)
    ema_value = exponential_smoothing(seq, window)[-1] # #####
    weighted_mean_value = weighted_mean(seq)
    count_over_mean = np.sum(np.array(seq>mean_value).astype(int))
    # group 2 相对值特征
    itl =  (1 if seq[0] < seq[-1] else -1)
    chg = (seq[-1] - seq[-2])
    pos = (1 if min_value == max_value else ((seq[-1] - min_value) - (max_value - seq[-1])) / (max_value - min_value))
    # group 3 tsf
    asc = absolute_sum_of_changes(seq)
    mac = mean_abs_change(seq)

    msdc = mean_second_derivative_central(seq)

    # final
    return [cur_value, max_value, min_value , mean_value, var_value, ema_value, weighted_mean_value, count_over_mean, itl, chg, pos,
            asc, mac, msdc] # 14维的特征



def extract_feature_window_v1(seq, window):
    """
    直接返回原始窗口里边的数据作为特征
    """
    return seq



def extract_feature_window_v2(seq, window):
    """
    基于窗口提取特征， 只提取当前点对应的窗口里边的特征
    1.  -当前值
    2.  -The index for the type of K-line (ITL) 窗口中的 第一个值 < 最后一个值 ？ 1：-1
    3.  -The change rate of transaction money compared to the previous trading day (当前值 - 最近值)/最近值
    4.  -当前值在窗口中的相对位置 ((pc (t) − pl (t)) − (ph (t) − pc (t))) / (ph (t) − pl (t))
    
    5.  窗口中的最小值
    6.  窗口中的最大值
    
    7.  -窗口内的均值
    8.  -窗口内的方差
    9.  -Exponential Moving Average list
    10. -weighted mean
    11. -count_over_mean
    """
    # group 1 绝对值特征
    max_value = np.max(seq)
    min_value = np.min(seq)
    cur_value = seq[-1]
    mean_value = np.mean(seq)
    var_value = np.var(seq)
    ema_value = exponential_smoothing(seq, window)[-1] # #####
    weighted_mean_value = weighted_mean(seq)
    count_over_mean = np.sum(np.array(seq>mean_value).astype(int))
    # group 2 相对值特征
    itl =  (1 if seq[0] < seq[-1] else -1)
    chg = (seq[-1] - seq[-2])
    pos = (1 if min_value == max_value else ((seq[-1] - min_value) - (max_value - seq[-1])) / (max_value - min_value))

    # final
    return [cur_value, max_value, min_value , mean_value, var_value, ema_value, weighted_mean_value, count_over_mean, itl, chg, pos] # 11维的特征
