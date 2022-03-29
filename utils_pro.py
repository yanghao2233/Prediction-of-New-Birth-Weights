# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, roc_auc_score, auc

import pandas as pd
import numpy as np

def sps2csv(spss_path, csv_path):
    raw = pd.read_spss(spss_path)
    raw.to_csv(csv_path)

def convert_one_hot(label_list):
    for i in range(len(label_list)):
        sec_col = np.ones([label_list[i].shape[0],label_list[i].shape[1],1])
        label_list[i] = np.reshape(label_list[i],[label_list[i].shape[0],label_list[i].shape[1],1])
        sec_col -= label_list[i]
        label_list[i] = np.concatenate([label_list[i],sec_col],2)
    return label_list

def load_pkl(file_path):
    """
    Pickle 文件读取函数，相对于 .csv 文件读取速度大幅增加。
        -> 在 preprocess 中进行 csv2pkl 的转换
    :param
        file_path: .pkl 文件的储存路径，格式为 string
    :return:
        base: 以 pkl 格式储存的数据， 格式为 list
    """
    with open(file_path) as f:
        base = pd.read_pickle(file_path)
    return base

def char2label(char_list):
    """
    字符串类型 与 整数 label 编码互转函数。
        @ 在当前任务暂无用武之地，备用
    :param
        char_list: 字符串列表，格式为 list
    :return:
        labels: 整数编码列表，格式为 list
    """
    char_encoder = LabelEncoder()
    labels = char_encoder.fit_transform(char_list)
    return labels

def label2onehot(label_list: list):
    """
    labels 与 onehot 形式互转的函数。便于后续使用。
    :param
        label_list: label列表，格式为 list 嵌套 list。list内含物格式不限。
            -> label_list = [[1], [2], [3]]
    :return:
        onehots: onehots格式的列表，格式为 list。list内为 binary 的类别码。
    """
    labels = np.array(label_list)
    binary = MultiLabelBinarizer()
    return binary.fit_transform(labels)

if __name__ == '__main__':
    t0 = np.array([[1, 2], [2, 5], [3], [1], [2], [5, 1, 3], [4]])
    print(label2onehot(t0))