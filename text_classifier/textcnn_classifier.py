# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 9:50
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : textcnn_classifier.py
# @Software: PyCharm
# DESC :TextCnn实现文本多分类，包含使用word2vector的预训练向量
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from process_data import load_data, class_dict
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
from sklearn import metrics
import gensim

logger = logging.getLogger(__name__)
maxlen = 64


# word2vector 词向量，此处为64维，可以使用其他训练好的词向量
def get_w2v():
    model_file = '../lib/word2vec_baike64/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True, limit=200000)
    # 预训练的词向量中没有出现的词用0向量表示
    embedding_matrix = np.zeros((len(vocab) + 1, 64))
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix



# 无使用预训练词向量的TextCnn
# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
def TextCnn1(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels):
    main_input = Input(shape=(maxlen,), dtype='float64')

    embedder = Embedding(len(vocab)+1, 256, input_length=maxlen, trainable=False)
    embed = embedder(main_input)

    # 三个卷积窗口为3,4,5
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)

    # 拼接-合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    # 全连接
    flat = Flatten()(cnn)

    # dropout
    drop = Dropout(0.2)(flat)

    # softmax
    main_output = Dense(15, activation='softmax')(drop)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.debug(model.summary())

    print('Train TextCnn...')
    model_weight_filepath = "./model/textcnn_model.weights"
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
    print('准确率', metrics.accuracy_score(y_test_labels, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test_labels, y_predict, average='weighted'))

    # 模型的保存
    # model.save('model.h5')

# 使用预训练词向量word2vector的TextCnn
# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
def TextCnn2(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels):
    main_input = Input(shape=(maxlen,), dtype='float64')

    embedding_matrix = get_w2v()
    embedder = Embedding(len(vocab) + 1, 64,  weights=[embedding_matrix], input_length=maxlen, trainable=False)
    embed = embedder(main_input)

    # 三个卷积窗口为3,4,5
    cnn1 = Convolution1D(64, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(64, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(64, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)

    # 拼接-合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    # 全连接
    flat = Flatten()(cnn)

    # dropout
    drop = Dropout(0.2)(flat)

    # softmax
    main_output = Dense(15, activation='softmax')(drop)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.debug(model.summary())

    print('Train TextCnn with Word2Vector...')
    model_weight_filepath = "./model/textcnnw2v_model.weights"
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


TextCnn1(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels)
# TextCnn2(x_train_padded_seqs, y_train_labels, x_test_padded_seqs, y_test_labels)

"""
TextCnn2 10轮：
    准确率 0.8489508479448117
    平均f1-score: 0.848463851184001

TextCnn1 10轮：
    准确率 0.8513810133528443
    平均f1-score: 0.8511511781649438

"""
