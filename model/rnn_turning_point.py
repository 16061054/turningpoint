import sys
sys.path.append('../')

import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras import losses, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from model.rnn import RNN, RNN_SEPARATE_2
from utils.custom_callback import Call_back_0, Call_back_1, Call_back_2
from model.loss import myloss


def _build_model(time_step, fearure_dim, modelName):
    i = Input(shape=(time_step, fearure_dim))
    if modelName == 'FEMT_LSTM':
        m = RNN_SEPARATE_2(time_step, fearure_dim)(i)
    else:
        m = RNN(time_step, fearure_dim)(i)
    m = Dense(1, activation='sigmoid')(m)
    model = Model(inputs=[i], outputs=[m])
    model.summary()
    # model.compile(optimizer=optimizers.Adam(lr=0.001, clipvalue=15), loss=focal_loss(alpha=0.75, gamma=0)) #alpha=0.75, gamma=0 Adam(lr=0.0001)
    model.compile(optimizer=optimizers.Adam(lr=0.001, clipvalue=15), loss=myloss(alpha=0.75, gamma=0)) #alpha=0.75, gamma=0 Adam(lr=0.0001)
    return model


def build_train(datas, machineID, modelName):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = datas

    num_samples = train_data.shape[0]
    time_step = train_data.shape[1]
    fearure_dim = train_data.shape[2]

    model = _build_model(time_step, fearure_dim, modelName)
    print('Train...')
    model_save_path = './lib/model_cp_rnn_{}'.format(machineID)

    call_backs = [Call_back_0(valid_data=[val_data, val_labels, test_data, test_labels], # test_data, test_labels
                              model_save_path=model_save_path),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, mode='min'), # 4
                  ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
                ]

    model.fit(x=train_data,
            y=train_labels,
            batch_size=64,
            epochs=60, # 30 60
            callbacks=call_backs,
            validation_data=[val_data, val_labels],
            )
    model.load_weights(model_save_path)
    return model


def build_test(datas, machineID, modelName):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = datas
    num_samples = train_data.shape[0]
    time_step = train_data.shape[1]
    fearure_dim = train_data.shape[2]
    model = _build_model(time_step, fearure_dim, modelName)
    print('Test...')
    model_save_path = './lib_baseline/lib_{}/model_cp_rnn_{}'.format(modelName, machineID)
    model.load_weights(model_save_path)
    return model