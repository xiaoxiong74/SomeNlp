# 文本分类实例集合

## Introduction
一些文本分类的示例，本项目所有示例是基于新闻的多分类(15类)，数据描述详见 [这里](https://github.com/xiaoxiong74/SomeNlp/tree/master/text_classifier/data)
* [基于SVM与贝叶斯的文本分类器](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_classifier/svm_and_bayes_classfier.py)
* [基于cnn的文本分类器](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_classifier/cnn_classifier.py)
* [基于rnn(lstm\gru)的文本分类器](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_classifier/rnn_classifier.py)
* [基于textcnn的文本分类器](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_classifier/textcnn_classifier.py)
* [基于bert的文本分类器](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_classifier/bert_classifier.py)

## Enviornment
python == 3.6  
keras == 2.2.4  
tensorflow == 1.12.0  
keras-bert == 0.71.0  

## Useage
* 下载数据集到data目录，分词后的数据使用的jieba分词
* 直接运行对应的分类器代码
