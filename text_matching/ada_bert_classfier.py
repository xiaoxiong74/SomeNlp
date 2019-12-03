# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/28 20:04
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : ada_bert_classfier.py
# @Software: PyCharm
# DESC : 基于bert的等价性问题判别
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

pd.set_option('display.max_columns', None)
maxlen=100

def save_data(is_cut=False):
    x_data = []
    y_data = []
    with open('./data/train_data.txt', "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp_line = line.split('\t')
            tmp_x = (tmp_line[0].strip(), tmp_line[1].strip())
            x_data.append(tmp_x)
            y_data.append(int(tmp_line[2].strip()))
    return x_data, y_data


# bert预训练模型路径
config_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../lib/bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'


token_dict = {}

class_numer = 2

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
                text1 = d[0][0]
                text2 = d[0][1]
                x1, x2 = tokenizer.encode(first=text1, second=text2)
                y = [d[1]]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    print(X1.shape, X2.shape)
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


def trian_model_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    print(x.shape)
    x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

# maxpool时消除mask部分的影响
def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return MaxPool1D(padding='same')(seq)
    # return K.max(seq, keepdims=True)

def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return AvgPool1D(padding='same')(seq)


def trian_model_bertlstmgru():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x1, x2 =x1_in, x2_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
    x = bert_model([x1, x2])
    t = Dropout(0.1)(x)
    t = Bidirectional(LSTM(80, recurrent_dropout=0.1, return_sequences=True))(t)
    t = Bidirectional(GRU(80, recurrent_dropout=0.1, return_sequences=True))(t)
    t = Dropout(0.4)(t)
    t = Dense(160)(t)
    # t_maxpool = Lambda(seq_maxpool)([t, mask])
    # t_maxpool = MaxPool1D()(t)
    # t_avgpool = Lambda(seq_avgpool)([t, mask])
    # t_ = concatenate([t_maxpool, t_avgpool], axis=-1)
    print(x.shape,  t.shape)
    # x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    c = concatenate([x, t], axis=-1)
    c = Lambda(lambda c: c[:, 0])(c)
    p = Dense(1, activation='sigmoid')(c)

    model = Model([x1, x2], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(2e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

def get_mode_type(model_type = "trian_model_bertlstmgru"):
    trian_model = ''
    if model_type == "trian_model_bert":
        trian_model = trian_model_bert()
    elif model_type == "trian_model_bertlstmgru":
        trian_model = trian_model_bertlstmgru()
    return trian_model, model_type


def fit_mode(train_data, valid_data, max_len ,i, epochs, batch_size):
    print(train_data[0])
    train_D = data_generator(train_data, max_len, batch_size)
    valid_D = data_generator(valid_data, max_len, batch_size)
    model, model_type = get_mode_type()
    model_weight_filepath = "./model/"+model_type+ str(i) + ".weights"
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
    reducelronplateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=epochs,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[earlystopping, reducelronplateau, checkpoint])
    del model
    K.clear_session()


def test(outfile="./data/dev_result.csv"):
    pbar = tqdm()
    dev_set = pd.read_csv("./data/dev_set.csv", encoding='utf-8', delimiter='\t')
    result = []
    result1 = []
    for index, row in dev_set.iterrows():
        content1 = str(row["question1"])
        content2 = str(row["question2"])
        x1, x2 = tokenizer.encode(first=content1, second=content2)
        x1 = x1[:maxlen]
        x2 = x2[:maxlen]
        tmp_result = model.predict([np.array([x1]), np.array([x2])])
        result_label = tmp_result[0][0]
        result1.append(result_label)  # 查看预测概率分布情况
        if result_label > 0.5:
            result_label = 1
        else:
            result_label = 0
        print(result_label)
        result.append(int(result_label))
        pbar.update(1)
    dev_set['label'] = pd.DataFrame(result, columns=['label'])
    dev_set['label'] = dev_set['qid'].astype(str) + "\t" + dev_set['label'].astype(str)
    dev_set = dev_set.drop(columns=['qid', 'question1', 'question2'])
    dev_set.to_csv(outfile, header=False, index=False)
    pbar.close()

tokenizer = OurTokenizer(token_dict)
train_x, train_y = save_data()
# train_y = [str(i) for i in train_y]


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
        fit_mode(train_data, valid_data, 100, i, 5, 16)
else:
    model, model_type = get_mode_type()
    model.load_weights("./model/bert_classfition_best_model0.weights")
    test()
