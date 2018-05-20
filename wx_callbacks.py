# coding:utf-8
from keras.callbacks import Callback


class MyCallBack(Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accs = []
        self.epoch_accs = []
        self.epoch_losses = []
        self.epoch_val_accs = []
        self.epoch_val_losses = []

    def on_batch_end(self, batch, logs={}):
        # loss,batch,acc,size
        self.batch_losses.append(logs.get('loss'))
        self.batch_accs.append(logs.get("acc"))
        # self.batch_val_accs.append(logs.get("val_acc"))
        # print(batch,logs)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # acc loss val_acc val_loss
        self.epoch_accs.append(logs.get("acc"))
        self.epoch_losses.append(logs.get("loss"))
        self.epoch_val_accs.append(logs.get("val_acc"))
        self.epoch_val_losses.append(logs.get("val_loss"))
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        print("losses:\n", self.batch_losses)
        print("batch_accs:\n", self.batch_accs)
        # print("batch_val_accs:\n", self.batch_val_accs)
        print("epoch_accs:\n", self.epoch_accs)
        print("epoch_losses:\n", self.epoch_losses)
        print("epoch_val_losses:\n", self.epoch_val_losses)
        print("epoch_val_accs:\n", self.epoch_val_accs)
        pass
