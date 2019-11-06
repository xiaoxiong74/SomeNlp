# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/6 10:04
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : svm_and_bayes_classfier.py
# @Software: PyCharm
# DESC 用传统机器学习方法 支持向量机SVM与朴素贝叶斯进行文本多分类
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from process_data import load_data, class_dict


x_data, y_data = load_data()
categories = list(class_dict.keys())
print(len(categories))
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=2019)

print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

"""朴素贝叶斯分类器"""
bayes_clf = Pipeline([("vect", CountVectorizer()),
                      ("tfidf", TfidfTransformer()),
                      ("clf", MultinomialNB())])

bayes_clf.fit(x_train, y_train)

bayes_predicted = bayes_clf.predict(x_test)
print('朴素贝叶斯分类器准确率：{:4.4f}'.format(np.mean(bayes_predicted == y_test)))
print(metrics.classification_report(y_test, bayes_predicted, target_names=categories))



"""SVM分类器"""
svm_clf = Pipeline([("vect", CountVectorizer()),
                      ("tfidf", TfidfTransformer()),
                      ("clf", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter= 5, random_state=42))])
svm_clf.fit(x_train, y_train)

svm_predicted = svm_clf.predict(x_test)
print('SVM分类器准确率：{:4.4f}'.format(np.mean(svm_predicted == y_test)))
print(metrics.classification_report(y_test, svm_predicted, target_names=categories))
# print("Confusion Matrix:")
# print(metrics.confusion_matrix(y_test, svm_predicted))
# print('\n')

""" 10-fold cross vaildation """
clf_b = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())
clf_s = make_pipeline(CountVectorizer(), TfidfTransformer(), SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))

bayes_5_fold = cross_val_score(clf_b, x_data, y_data, cv=5)
svm_5_fold = cross_val_score(clf_s, x_data, y_data, cv=5)

print('Naives Bayes 5-fold correct prediction: {:4.4f}'.format(np.mean(bayes_5_fold)))
print('SVM 5-fold correct prediction: {:4.4f}'.format(np.mean(svm_5_fold)))


"""
最终得到的准确率：
朴素贝叶斯分类器准确率：0.8699
SVM分类器准确率：0.8470
Naives Bayes 5-fold correct prediction: 0.8589
SVM 5-fold correct prediction: 0.8359
"""