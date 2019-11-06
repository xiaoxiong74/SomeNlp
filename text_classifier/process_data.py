# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 9:39
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : process_data.py
# @Software: PyCharm
# DESC :处理今日头条数据
数据来源：https://github.com/fate233/toutiao-text-classfication-dataset
"""

import numpy as np
import re
import jieba

"""
文本类别，共15类
100 故事 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
"""

class_dict = {
    '100':  'news_story',
    '101':  'news_culture',
    '102':  'news_entertainment',
    '103':  'news_sports',
    '104':  'news_finance',
    '106':  'news_house',
    '107':  'news_car',
    '108':  'news_edu ',
    '109':  'news_tech',
    '110':  'news_military',
    '112':  'news_travel',
    '113':  'news_world',
    '114':  'stock',
    '115':  'news_agriculture',
    '116':  'news_game'
}


class_dict_map = {
    "0": '100',
    "1": '101',
    "2": '102',
    "3": '103',
    "4": '104',
    "5": '106',
    "6": '107',
    "7": '108',
    "8": '109',
    "9": '110',
    "10": '112',
    "11": '113',
    "12": '114',
    "13": '115',
    "14": '116'
}

# 加载时处理数据
def save_data(is_cut=False):
    punctuation_remove = '[：；……（）『』《》【】★～!"#$%&\'()*+,-./:;<=>？！“”：、，。?@[\\]^_`{|}~]+'
    # cw = lambda x: list(jieba.cut(x))
    x_data = []
    y_data = []
    with open('./data/toutiao_cat_data.txt', "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp_line = line.split('_!_')
            tmp_x = re.sub(punctuation_remove, '', tmp_line[3].strip()+tmp_line[4].strip())
            x_data.append(' '.join(tmp_x.split()).strip())
            y_data.append(tmp_line[1].strip())
    if is_cut:
        x_data = list(map(lambda x: ' '.join(list(jieba.cut(x))), x_data))
    return x_data, y_data


# 先分词保存，每次使用时直接使用分词后的数据
def load_data():
    x_data = []
    y_data = []
    with open('./data/toutiao_cat_data_cut.txt', "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp_line = line.split(',')
            x_data.append(tmp_line[0])
            tmp_y = re.sub(r'\n', '', tmp_line[1])
            y_data.append(tmp_y)
    return x_data, y_data


