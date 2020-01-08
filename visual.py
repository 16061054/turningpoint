import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,4)
np.set_printoptions(suppress = True)


from utils.dataSet import Indicator


def show_labeled_data(machineID):
    ind = Indicator(window=10, window_list=[3], machineID = '207776314', isRaw=False, isCurr=True)
    labels = ind.data_df['y'].values
    data = ind.data_df['cpu'].values
    print("ind.count_t: ", ind.count_t)
    print("ind.count_s: ", ind.count_s)
    
    nn = 200 # 画出前200个点的峰值和谷值情况
    plt.plot(list(range(len(data)))[:nn], data[:nn], color='blue')
    marker_list = ['o', 'v']
    color_list = ['blue','red']
    label_des_list = ['common point', 'turning point']

    for i in range(len(marker_list)):
        ids = np.where(labels==i)[0]
        ii = np.sum(np.array(ids < nn).astype(int))
        plt.scatter(ids[:ii], np.array(data)[ids][:ii], marker=marker_list[i], color=color_list[i], label=label_des_list[i])
        plt.xlabel("time/5min")
        plt.ylabel("cpu usage rate")
    plt.legend()
    plt.show()


def show_predict(machineID, start_ids=300, end_ids=400, isRaw=False, window=10, modelName="FEMT_LSTM", check=False,):
    predict_result_file_path = './lib_baseline/lib_{}/model_pred_rnn_{}.npy'.format(modelName, machineID)
    
    ind = Indicator(window, window_list=[3], machineID = machineID, isRaw=isRaw, isCurr=True) # 207776314(M1), 908054(M2), 1273805(M3）
    splited_data = ind.splited_data

    train_data, train_labels = splited_data["train_data"], splited_data["train_labels"]
    test_data, test_labels = splited_data["test_data"], splited_data["test_labels"]

    test_labels = test_labels.reshape(-1,)
    pred_labels = np.load(predict_result_file_path)
    data = test_data[:, -1 , 0]
    print('')
    print("test_labels.shape=", test_labels.shape)
    print("pred_labels.shape=", pred_labels.shape)
    print("test_data.shape=", data.shape)
    
    if check == True:
        check = ((pred_labels == test_labels).reshape(-1,) * (test_labels == 1).reshape(-1,)).astype(int)
        ids = np.where(check == 1)[0]
        print('')
        print("ture positive index in test set:")
        print(ids)
        print("ture positive count:", len(ids))
        # assert 1 == 0, "check done! please turn off arg : check and select start_ids and end_ids"

    test_data_len = len(test_data)
    if start_ids >=0 and start_ids < end_ids and end_ids < test_data_len:

        plt.plot(list(range(len(data)))[start_ids : end_ids], data[start_ids: end_ids], color='blue')
        marker_list = ['o', 'v']
        color_list = ['blue','red']
        label_des_list = ['predict common point', 'predict turning point']

        print("\ninputs:")
        for sampleids in range(start_ids, end_ids):
            print(test_data[sampleids])
        print("predict/true:")
        for sampleids in range(start_ids, end_ids):
            print(int(pred_labels[sampleids]), int(test_labels[sampleids]))

        for i in range(len(marker_list)):
            ids = np.where(pred_labels==i)[0]
            valided_ids_index = np.where((ids >= start_ids) & (ids < end_ids))[0]
            valided_ids = ids[valided_ids_index]
            plt.scatter(valided_ids, data[valided_ids], marker=marker_list[i], color=color_list[i], label=label_des_list[i])
            if i == 1:
                test_valided_ids = [i for i in valided_ids if test_labels[i]==1]
                print('')
                print("ture positive count in range {}-{}:".format(start_ids, end_ids), len(test_valided_ids))
                plt.scatter(test_valided_ids, data[test_valided_ids], color='', marker='o', edgecolors='g', s=80, label='true turning point')

        plt.xlabel("time/5min")
        plt.ylabel("cpu usage rate")
        plt.legend()
        plt.show()
    else:
        print("Invalid start_ids and end_ids!")


if __name__ == '__main__':
    # test show_labeled_data
    machineID = '207776314'
    show_labeled_data('207776314')
    # test show_predict
    start_ids = 300
    end_ids = 400
    isRaw = False
    window = 10
    show_predict(machineID, start_ids, end_ids, isRaw, window)