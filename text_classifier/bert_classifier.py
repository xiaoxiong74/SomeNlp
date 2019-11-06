# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 10:07
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : bert_classifier.py
# @Software: PyCharm
# DESC : 基于bert的文本分类
"""

import codecs
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from sklearn.metrics import f1_score
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tqdm import tqdm
from process_data import save_data, class_dict_map


pd.set_option('display.max_columns', None)
# 输入的样本信息的最大长度
# max_len = 128
config_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

token_dict = {}

class_dict = {i: j for j, i in class_dict_map.items()}
print(class_dict)
class_numer = len(class_dict)

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, max_len, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.max_len = max_len
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.max_len]
                #text = list(d[0])
                x1, x2 = tokenizer.encode(first=text)
                y = [class_dict[d[1]]]
                # y = class_dict.get(d[1], '0')
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# 样本不均衡时使用的损失函数focal_loss
def focal_loss_fixed(y_true, y_pred):
    # y_pred = K.sigmoid(y_pred)
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-6
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + epsilon))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0  + epsilon))


def trian_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    # x = Dropout(0.2)(x)
    # x = Bidirectional(LSTM(128, recurrent_dropout=0.1, return_sequences=True))(x)
    # x = Dropout(0.2)(x)
    print(x.shape)
    x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    p = Dense(class_numer, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


def fit_mode(train_data, valid_data, max_len ,i, epochs, batch_size):
    print(train_data[0])
    train_D = data_generator(train_data, max_len, batch_size)
    valid_D = data_generator(valid_data, max_len, batch_size)
    model = trian_model()
    model_weight_filepath = "./model/bert_classfition_best_model" + str(i) + ".weights"
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
    reducelronplateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=epochs,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[earlystopping, reducelronplateau, checkpoint])
    del model
    K.clear_session()


def test():
    pass

tokenizer = OurTokenizer(token_dict)
train_x, train_y = save_data()
train_y = [str(i) for i in train_y]

if __name__ == '__main__':
    # 多次shuffe，k-fold
    for i in range(0, 5):
        data = []
        for d, y in zip(train_x, train_y):
            data.append((d, y))
        # 按照9:1的比例划分训练集和验证集
        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
        valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
        fit_mode(train_data, valid_data, 160, i, 10, 32)
else:
    test()
