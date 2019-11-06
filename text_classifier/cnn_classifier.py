# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 9:24
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : cnn_classifier.py
# @Software: PyCharm
# DESC : CNN实现文本多分类
"""

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import  BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from process_data import load_data, class_dict
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
from sklearn import metrics


logger = logging.getLogger(__name__)
maxlen = 64
batch_size = 512

# 当数据集较大时使用批量迭代生成器
# def data_generator(data, targets, batch_size):
#     batches = (len(data) + batch_size - 1)//batch_size
#     while(True):
#          for i in range(batches):
#               X = data[i*batch_size: (i+1)*batch_size]
#               Y = targets[i*batch_size: (i+1)*batch_size]
#               x_wordids = tokenizer.texts_to_sequences(X)
#               x_sequence= tokenizer.sequences_to_matrix(x_wordids, mode='binary')
#               yield (x_sequence, Y)


#构建CNN分类模型(LeNet-5)
#模型结构：嵌入-卷积池化*2-dropout-BN-全连接-dropout-全连接
def Cnn_Model(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 256, input_length=maxlen))
    model.add(Convolution1D(256, 5, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(128, 5, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # (批)规范化层
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    logger.debug(model.summary())

    print('Train Cnn_Model...')
    model_weight_filepath = "./model/cnn_model.weights"
    checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=4)
    model.fit(x_train_padded_seqs, y_train_labels,
              batch_size=512,
              epochs=10,
              validation_data=(x_test_padded_seqs, y_test_labels),
              callbacks=[earlystopping, checkpoint])

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))

    y_test_labels = np.argmax(y_test_labels, axis=1)
    y_test_labels = list(map(str, y_test_labels))
    print('准确率', metrics.accuracy_score(y_test_labels, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test_labels, y_predict, average='weighted'))

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


Cnn_Model(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels)

"""
Cnn_Model 10轮：
    准确率 0.8636102328255246
    平均f1-score: 0.8628608729762441
"""