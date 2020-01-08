import sys

from baseline_lr_basic import main as lr_basic_test
from baseline_lr_fluctuant import main as lr_fluctuant_test
from baseline_svm_basic import main as svm_basic_test
from baseline_svm_fluctuant import main as svm_fluctuant_test
from baseline_main_rnn import test_rnn_turning_point
from visual import show_labeled_data, show_predict
from utils.segmentation import main as segment_func

# 207776314(M1), 908054(M2), 1273805(M3)
# python main.py M1 0 FEMT_LSTM

print("valid dataSetID: M1,M2,M3")
dataSetID = input("enter dataSet:\n")

# print("cmd format:  python main.py dataSetID showlabeledData modelName")
# print("cmd example: python  main.py M1 0 FEMT_LSTM")

showSegment = input("show segmentation procedure? enter 1 or 0\n")
if dataSetID == 'M1':
    if showSegment == "1":
        while True:
            max_error = float(input("enter max_error for segment, default 0.0015, enter 0 for jump out:\n"))
            if max_error == 0:
                break
            segment_func('207776314', max_error, TD=1)

    showlabeledData = input("show labeled data? enter 1 or 0\n")
    if showlabeledData == '1':
        show_labeled_data('207776314')

    print("valid model name: lb,lf,sb,sf,FEMT_LSTM,S_LSTM,P_LSTM")
    modelName = input("enter model name:\n")

    if modelName == 'lb':
        print("current model lr_basic:")
        lr_basic_test(window_list=[3], machineID='207776314', isTrain=False)
    elif modelName == 'lf':
        print("current model lr_fluctuant:")
        lr_fluctuant_test(window_list=[3], machineID='207776314', isTrain=False)
    elif modelName == 'sb':
        print("current model svm_basic:")
        svm_basic_test(window_list=[3], machineID='207776314', isTrain=False)
    elif modelName == 'sf':
        print("current model svm_fluctuant:")
        svm_fluctuant_test(window_list=[3], machineID='207776314', isTrain=False)
    elif modelName == 'FEMT_LSTM':
        print("current model FEMT_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='207776314', isRaw=False, isCurr=True, modelName=modelName)
        showFEMTLSTMPredictedLabel = input("do you want to see FEMT_LSTM predict result? enter 1 or 0\n")
        if showFEMTLSTMPredictedLabel == '1':
            while True:
                start_ids = int(input("input start index in test set you want to see, enter -1 to quit:\n"))
                if start_ids == -1:
                    break
                end_ids = int(input("input end index in test set you want to see:\n"))
                show_predict('207776314', start_ids, end_ids, False, 10)
    elif modelName == 'S_LSTM':
        print("current model S_LSTM:")
        test_rnn_turning_point(window=5, window_list=[3], machineID='207776314', isRaw=False, isCurr=True, modelName=modelName)
    elif modelName == 'P_LSTM':
        print("current model P_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='207776314', isRaw=True, isCurr=True, modelName=modelName)
    else:
        print("{} are not valid model name!".format(modelName))
        assert 1 == 0, "modelName error!"


elif dataSetID == 'M2':
    if showSegment == "1":
        while True:
            max_error = float(input("enter max_error for segment, default 0.0015, enter 0 for jump out:\n"))
            if max_error == 0:
                break
            segment_func('908054', max_error, TD=1)
    
    showlabeledData = input("show labeled data? enter 1 or 0\n")
    if showlabeledData == '1':
        show_labeled_data('908054')

    print("valid model name: lb,lf,sb,sf,FEMT_LSTM,S_LSTM,P_LSTM")
    modelName = input("enter model name:\n")

    if modelName == 'lb':
        print("current model lr_basic:")
        lr_basic_test(window_list=[3], machineID='908054', isTrain=False)
    elif modelName == 'lf':
        print("current model lr_fluctuant:")
        lr_fluctuant_test(window_list=[3], machineID='908054', isTrain=False)
    elif modelName == 'sb':
        print("current model svm_basic:")
        svm_basic_test(window_list=[3], machineID='908054', isTrain=False)
    elif modelName == 'sf':
        print("current model svm_fluctuant:")
        svm_fluctuant_test(window_list=[3], machineID='908054', isTrain=False)
    elif modelName == 'FEMT_LSTM':
        print("current model FEMT_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='908054', isRaw=False, isCurr=True, modelName=modelName)
        showFEMTLSTMPredictedLabel = input("do you want to see FEMT_LSTM predict result? enter 1 or 0\n")
        if showFEMTLSTMPredictedLabel == '1':
            while True:
                start_ids = int(input("input start index in test set you want to see, enter -1 to quit:\n"))
                if start_ids == -1:
                    break
                end_ids = int(input("input end index in test set you want to see:\n"))
                show_predict('908054', start_ids, end_ids, False, 10)
    elif modelName == 'S_LSTM':
        print("current model S_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='908054', isRaw=False, isCurr=True, modelName=modelName)
    elif modelName == 'P_LSTM':
        print("current model P_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='908054', isRaw=True, isCurr=True, modelName=modelName)
    else:
        print("{} are not valid model name!".format(modelName))
        assert 1 == 0, "modelName error!"


elif dataSetID == 'M3':
    if showSegment == "1":
        while True:
            max_error = float(input("enter max_error for segment, default 0.0015, enter 0 for jump out:\n"))
            if max_error == 0:
                break
            segment_func('1273805', max_error, TD=1)
    
    showlabeledData = input("show labeled data? enter 1 or 0\n")
    if showlabeledData == '1':
        show_labeled_data('1273805')
    
    print("valid model name: lb,lf,sb,sf,FEMT_LSTM,S_LSTM,P_LSTM")
    modelName = input("enter model name:\n")

    if modelName == 'lb':
        print("current model lr_basic:")
        lr_basic_test(window_list=[3], machineID='1273805', isTrain=False)
    elif modelName == 'lf':
        print("current model lr_fluctuant:")
        lr_fluctuant_test(window_list=[3], machineID='1273805', isTrain=False)
    elif modelName == 'sb':
        print("current model svm_basic:")
        svm_basic_test(window_list=[3], machineID='1273805', isTrain=False)
    elif modelName == 'sf':
        print("current model svm_fluctuant:")
        svm_fluctuant_test(window_list=[3], machineID='1273805', isTrain=False)
    elif modelName == 'FEMT_LSTM':
        print("current model FEMT_LSTM:")
        test_rnn_turning_point(window=5, window_list=[3], machineID='1273805', isRaw=False, isCurr=True, modelName=modelName)
        showFEMTLSTMPredictedLabel = input("do you want to see FEMT_LSTM predict result? enter 1 or 0\n")
        if showFEMTLSTMPredictedLabel == '1':
            while True:
                start_ids = int(input("input start index in test set you want to see, enter -1 to quit:\n"))
                if start_ids == -1:
                    break
                end_ids = int(input("input end index in test set you want to see:\n"))
                show_predict('908054', start_ids, end_ids, False, 5)
    elif modelName == 'S_LSTM':
        print("current model S_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='1273805', isRaw=False, isCurr=True, modelName=modelName)
    elif modelName == 'P_LSTM':
        print("current model P_LSTM:")
        test_rnn_turning_point(window=10, window_list=[3], machineID='1273805', isRaw=True, isCurr=True, modelName=modelName)
    else:
        print("{} are not valid model name!".format(modelName))
        assert 1 == 0, "modelName error!"


else:
    print("dataSetID must be:")
    print("M1 or M2 or M3")
    assert 1 == 0, "dataSetID error!"