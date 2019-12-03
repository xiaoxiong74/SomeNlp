# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/28 17:46
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : data_process.py
# @Software: PyCharm
# DESC :
"""
import sys
import importlib
import re
import jieba
import codecs
import json
importlib.reload(sys)

from xml.dom.minidom import parse
def generate_train_data_pair(equ_questions, not_equ_questions):
    a = [x+"\t"+y+"\t"+"0" for x in equ_questions for y in not_equ_questions]
    b = [x+"\t"+y+"\t"+"1" for x in equ_questions for y in equ_questions if x!=y]
    return a+b
def parse_train_data(xml_data):
    pair_list = []
    doc = parse(xml_data)
    collection = doc.documentElement
    for i in collection.getElementsByTagName("Questions"):
        # if i.hasAttribute("number"):
        #     print ("Questions number=", i.getAttribute("number"))
        EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
        NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
        equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
        not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
        equ_questions_list, not_equ_questions_list = [], []
        for q in equ_questions:
            try:
                equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        for q in not_equ_questions:
            try:
                not_equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        pair = generate_train_data_pair(equ_questions_list, not_equ_questions_list)
        pair_list.extend(pair)
    print("All pair count=", len(pair_list))
    return pair_list
def write_train_data(file, pairs):
    with open(file, "w", encoding='utf-8') as f:
        for pair in pairs:
            f.write(pair+"\n")

def cut_word(file, cut_file):
    tmp_list = []
    punctuation_remove = '[：；……（）『』《》【】★～!"#$%&\'()*+,-./:;<=>？！“”：、，。?@[\\]^_`{|}~]+'
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp_line = line.split('\t')
            x1 = ' '.join(list(jieba.cut(re.sub(punctuation_remove, '', tmp_line[0].strip()))))
            x2 = ' '.join(list(jieba.cut(re.sub(punctuation_remove, '', tmp_line[1].strip()))))
            x3 = tmp_line[2].strip()
            x = x1.strip()+","+x2.strip()+","+x3
            tmp_list.append(x)
    write_train_data(cut_file, tmp_list)

def all_chars():
    chars = {}
    with open("./data/train_data.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            for word in line:
                chars[word] = chars.get(word, 0) + 1
    with open("./data/dev_set.csv", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            for word in line:
                chars[word] = chars.get(word, 0) + 1
    return chars
if __name__ == "__main__":
    min_count = 2
    pair_list = parse_train_data("./data/train_set.xml")
    write_train_data("./data/train_data.txt", pair_list)
    cut_word("./data/train_data.txt", "./data/train_data_cut.txt")
    chars = all_chars()
    with codecs.open('./data/all_chars.json', 'w', encoding='utf-8') as f:
        chars = {i: j for i, j in chars.items() if j >= min_count}
        id2char = {i + 2: j for i, j in enumerate(chars)}  # padding: 0, unk: 1
        char2id = {j: i for i, j in id2char.items()}
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
