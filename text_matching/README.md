# 文本匹配实例

## Introduction
本文主要为短文本匹配的简单实例demo，更多思路可参考 [这里](https://github.com/xiaoxiong74/SomeNlp/blob/master/some_solutions)  
本文数据来源 [地址](https://biendata.com/competition/2019diac/)，可注册下载或留邮箱发送
* [基于bert+bilstm的文本匹配](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_matching/ada_bert_classfier.py)
* [基于ESIM的短文本匹配](https://github.com/xiaoxiong74/SomeNlp/blob/master/text_matching/esim.py)


## Enviornment
python == 3.6  
keras == 2.2.4  
tensorflow == 1.12.0  
keras-bert == 0.71.0  

## Useage
* 下载数据集到data目录，分词后的数据使用的jieba分词
* 先运行 python data_process.py
* 直接运行 python ada_bert_classfier.py   或者 python esim.py 运行对应的匹配模型