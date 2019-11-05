# 关系抽取实例
## Introduction
本demo中的所有关系抽取实例均参考与于苏神的科学空间：https://spaces.ac.cn/
## Enviornment
python == 3.6  
keras == 2.2.4  
tensorflow == 1.12.0  
keras-bert == 0.71.0  
## Useage
* 下载对应的数据与预训练数据到对应的文件，其中lib中文预训练数据，data中训练数据，下载方式详见对应文件夹
* 运行data_trans.py文件处理为模型的输入数据
* dgcnn_without_ds.py 为基于门膨胀卷积的三元组抽取
* dgcnn_with_ds.py 为加入部分先验信息的门膨胀卷积的三元组抽取
* bert_relation_extract.py 为基于bert的三元组抽取