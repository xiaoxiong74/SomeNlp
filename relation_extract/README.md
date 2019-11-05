# 关系抽取实例
## Introduction
信息抽取(Information Extraction, IE)是从自然语言文本中抽取实体、属性、关系及事件等事实类信息的文本处理技术，
是信息检索、智能问答、智能对话等人工智能应用的重要基础。  
示例数据：
```
    {
        "text": "《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧",
        "spo_list": [
            [
                "娘家的故事第二部",
                "导演",
                "张玲"
            ],
            [
                "娘家的故事第二部",
                "主演",
                "林在培"
            ],
            [
                "娘家的故事第二部",
                "主演",
                "何赛飞"
            ]
        ]
    }
```
输入一个句子，然后输出该句子包含的所有三元组。

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