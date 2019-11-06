# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 9:47
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : rnn_classifier.py
# @Software: PyCharm
# DESC Rnn模型实现文本多分类，包含lstm、gru
"""

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import LSTM, CuDNNLSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from process_data import load_data, class_dict
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
from sklearn import metrics

logger = logging.getLogger(__name__)
maxlen = 64


# Lstm
# 模型结构：词嵌入-LSTM-LSTM-全连接
def LstmModel(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 256, input_length=maxlen))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.debug(model.summary())

    print('Train LstmModel...')
    model_weight_filepath = "./model/Lstm_model.weights"
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
    model.fit(x_train_padded_seqs, y_train_labels,
              batch_size=256,
              epochs=100,
              validation_data=(x_test_padded_seqs, y_test_labels),
              callbacks=[earlystopping, checkpoint])

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)   # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))

    y_test_labels = np.argmax(y_test_labels, axis=1)
    y_test_labels = list(map(str, y_test_labels))
    print('LstmModel 准确率', metrics.accuracy_score(y_test_labels, y_predict))
    print('LstmModel 平均f1-score:', metrics.f1_score(y_test_labels, y_predict, average='weighted'))

    # 模型的保存
    # model.save('model.h5')


# Gru
# 模型结构：词嵌入-Gru-Gru-全连接
def GruModel(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 256, input_length=maxlen))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.debug(model.summary())

    print('Train GruModel...')
    model_weight_filepath = "./model/Gru_model.weights"
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
    model.fit(x_train_padded_seqs, y_train_labels,
              batch_size=256,
              epochs=100,
              validation_data=(x_test_padded_seqs, y_test_labels),
              callbacks=[earlystopping, checkpoint])

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))

    y_test_labels = np.argmax(y_test_labels, axis=1)
    y_test_labels = list(map(str, y_test_labels))
    print('GruModel准确率', metrics.accuracy_score(y_test_labels, y_predict))
    print('GruModel平均f1-score:', metrics.f1_score(y_test_labels, y_predict, average='weighted'))

    # 模型的保存
    # model.save('model.h5')

x_data, y_data = load_data()
categories = list(class_dict.keys())
print(len(categories))

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
# fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
tokenizer.fit_on_texts(x_data)
#得到每个词的编号
vocab = tokenizer.word_index
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2019)

print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
print(len(set(y_train)),len(set(y_test)))
# 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
x_train_wordids = tokenizer.texts_to_sequences(x_train)
x_test_wordids = tokenizer.texts_to_sequences(x_test)

# 序列模式
# 每条样本长度不唯一，将每条样本的长度设置一个固定值
x_train_padded_seqs = pad_sequences(x_train_wordids, maxlen=maxlen)
x_test_padded_seqs = pad_sequences(x_test_wordids, maxlen=maxlen)

# 编码类别
encoder = LabelEncoder()
encoded_y_train = encoder.fit_transform(y_train)
y_train_labels = to_categorical(encoded_y_train)

encoded_y_train = encoder.fit_transform(y_test)
y_test_labels = to_categorical(encoded_y_train)


# LstmModel(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels)
GruModel(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels)

"""
LstmModel 10轮：
    LstmModel 准确率 0.8778907209490711
    LstmModel 平均f1-score: 0.8778844698421312
    
GruModel 10轮：
    GruModel准确率 0.8799158587890982
    GruModel平均f1-score: 0.8796885377584901

"""

