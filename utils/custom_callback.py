from keras.callbacks import Callback
from sklearn.metrics import f1_score
import keras.backend as K


class Call_back_0(Callback):
    """
    每个epoch打印 val_f1
    """
    def __init__(self, valid_data, model_save_path=None):
        super(Call_back_0, self).__init__()
        self.X = valid_data[0]
        self.y = valid_data[1]

        self.X_test = valid_data[2]
        self.y_test = valid_data[3]

        self.model_save_path = model_save_path
        self.best_f1 = -1


    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_pred = (y_pred > 0.5).astype(int)
        f1 = f1_score(self.y, y_pred, average="binary")

        y_pred_test = self.model.predict(self.X_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        f1_test = f1_score(self.y_test, y_pred_test, average="binary")
        lr = float(K.get_value(self.model.optimizer.lr))

        print(">Epoch-%d || f1_val-%.4f || f1_test-%.4f || lr-%.4f" % (epoch, f1, f1_test, lr))


class Call_back_1(Callback):
    """
    每个epoch打印 val_f1, 并保存val_f1最大的模型
    """
    def __init__(self, valid_data, model_save_path=None):
        super(Call_back_1, self).__init__()
        self.X = valid_data[0]
        self.y = valid_data[1]

        self.X_test = valid_data[2]
        self.y_test = valid_data[3]

        self.model_save_path = model_save_path
        self.best_f1 = -1


    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_pred = (y_pred > 0.5).astype(int)
        f1 = f1_score(self.y, y_pred, average="binary")

        y_pred_test = self.model.predict(self.X_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        f1_test = f1_score(self.y_test, y_pred_test, average="binary")
        lr = float(K.get_value(self.model.optimizer.lr))

        print(">Epoch-%d || f1_val-%.4f || f1_test-%.4f || lr-%.4f" % (epoch, f1, f1_test, lr))

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights(self.model_save_path)



class Call_back_2(Callback):
    """
    按照val_f1进行学习率的衰减
    """
    def __init__(self, valid_data, model_save_path=None, wait_cnt=4, decay_rate=0.8, start_epoch=3):
        super(Call_back_2, self).__init__()
        self.X = valid_data[0]
        self.y = valid_data[1]

        self.X_test = valid_data[2]
        self.y_test = valid_data[3]

        self.model_save_path = model_save_path

        self.wait_cnt = wait_cnt

        self.decay_rate = decay_rate

        self.start_epoch = start_epoch

        self.best_f1 = -1

        self.wait_num = 0


    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_pred = (y_pred > 0.5).astype(int)
        f1 = f1_score(self.y, y_pred, average="binary")

        y_pred_test = self.model.predict(self.X_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        f1_test = f1_score(self.y_test, y_pred_test, average="binary")

        lr = float(K.get_value(self.model.optimizer.lr))
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights(self.model_save_path)
        else:
            if epoch >= self.start_epoch:
                self.wait_num += 1
                if self.wait_num >= self.wait_cnt:
                    lr = self.decay_rate * lr
                    K.set_value(self.model.optimizer.lr, lr)
                    self.wait_num = 0

        print(">Epoch-%d || f1_val-%.4f || f1_test-%.4f || lr-%.4f" % (epoch, f1, f1_test, lr))
