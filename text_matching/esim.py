# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/2 19:34
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : esim.py
# @Software: PyCharm
# DESC : 基于Enhanced LSTM for Natural Language Inference(ESIM)的等价性问题判别
"""


import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from gensim.models import Word2Vec
import jieba
import re
import json

pd.set_option('display.max_columns', None)
punctuation_remove = '[：；……（）『』《》【】★～!"#$%&\'()*+,-./:;<=>？！“”：、，。?@[\\]^_`{|}~]+'
maxlen=40
word2vec = Word2Vec.load('../lib/word2vec_baike/word2vec_baike')

id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])

id2char, char2id = json.load(open('./data/all_chars.json', encoding='utf-8'))


def save_data():
    """
    获取训练数据
    :return:
    """
    x_data = []
    y_data = []
    with open('./data/train_data_cut.txt', "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp_line = line.split(',')
            tmp_x = (tmp_line[0].strip(), tmp_line[1].strip())
            x_data.append(tmp_x)
            y_data.append(int(tmp_line[2].strip()))
    return x_data, y_data


def sent2vec(S):
    """
    词向量化
    S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


def tokenizer(x):
    """
    分词
    :param x:
    :return:
    """
    return list(jieba.cut(re.sub(punctuation_remove, '', x.strip())))


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
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
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
                x1 = tokenizer(d[0][0][:maxlen])
                x2 = tokenizer(d[0][1][:maxlen])
                y = [d[1]]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    # 词向量化
                    X1 = sent2vec(X1)
                    X2 = sent2vec(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def esim_model():
    """
    搭建esim 模型
    :return:
    """
    i1 = Input(shape=(None, word_size), dtype='float32')
    i2 = Input(shape=(None, word_size), dtype='float32')

    # input encoding
    x1, x2 = i1, i2

    x1 = Bidirectional(LSTM(256, return_sequences=True))(x1)
    x2 = Bidirectional(LSTM(256, return_sequences=True))(x2)

    # local inference modeling
    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis': 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.sum, arguments={'axis': 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis': 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.sum, arguments={'axis': 1})(_x2)

    # inference composition
    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = Bidirectional(LSTM(256, return_sequences=True))(m1)
    y2 = Bidirectional(LSTM(256, return_sequences=True))(m2)

    mx1 = Lambda(K.max, arguments={'axis': 1})(y1)
    av1 = Lambda(K.mean, arguments={'axis': 1})(y1)
    mx2 = Lambda(K.max, arguments={'axis': 1})(y2)
    av2 = Lambda(K.mean, arguments={'axis': 1})(y2)

    y = Concatenate()([av1, mx1, av2, mx2])
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[i1, i2], outputs=y)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(5e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


def fit_mode(train_data, valid_data ,i, epochs, batch_size):
    print(train_data[0])
    train_D = data_generator(train_data, batch_size)
    valid_D = data_generator(valid_data, batch_size)
    model = esim_model()
    model_weight_filepath = "./model/esim_"+ str(i) + ".weights"
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

train_x, train_y = save_data()

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
        fit_mode(train_data, valid_data, i, 30, 32)

