# -*- coding:utf-8 -*-
# @Author: yanghao
# @Create time: 2022/02/09 09:44
# @Description:
import logging
import numpy as np
import pandas as pd
import config
import pickle
from collections import Counter
import utils_pro
import logging as logger
from tqdm import tqdm
import tensorflow as tf

def get_label_pkl(file_path):
    """
    获取 onehot形式的 label, 以
    :param file_path:
    :return:
    """
    base = pd.read_csv(file_path)
    labels = base.iloc[:, -1].to_list()
    lbs = []

    for label in labels:
        lb = []
        if pd.isnull(label):
            label = 2
        else:
            label = int(label)
        lb.append(label)
        lbs.append(lb)
    batch_ls = utils_pro.label2onehot(lbs)

def get_data_slice(file_path, time_pkl_path = None, data_pkl_path = None, label_pkl_path = None, number = 5):
    """
    获取需要的切片数据, 直接输出为 pkl
    :param file_path:
    :return:
    """
    base = pd.read_csv(file_path)
    names = base.iloc[:, 0].to_list()
    counts = Counter(names)
    usable_list = []
    if number:
        for name, count in counts.items():
            if count == number:
                usable_list.append(name)
    logger.info('The whole dataset contains %s samples' % len(usable_list))
    # 集成处理备用 未实现
    # else:
    #     for count_ in range(5, 13):
    #         temp_list = []
    #         for name, count in counts.items():
    #             if count == count_:
    #                 temp_list.append(name)
    #         usable_list.append(temp_list)
    logger.info('****** Data Slice Process Initiate ******')
    usable_list = tqdm(usable_list)
    datas, times, labels = [], [], []
    for name in usable_list:
        single = single_locator(data = base, name = name)
        data = np.array(single.iloc[:, 3:10])
        time = []
        time_i = np.array(single.iloc[:, 1])
        time.append(time_i)
        # 集成了原有的 get_label_pkl 函数
        label = list(set(single.iloc[:, -1].to_list()))[0]
        if pd.isnull(label):
            label = 2
        else:
            label = int(label)
        labels.append([label])
        datas.append(data)
        times.append(time)

    labels = utils_pro.label2onehot(labels)
    # print(labels)
    times = tf.convert_to_tensor(times)
    # print(times)
    datas = tf.convert_to_tensor(datas)
    # print(datas)
    labels = tf.convert_to_tensor(labels)
    logger.info('****** Data Slice Finish ******')
    if time_pkl_path:
        with open(time_pkl_path, 'wb') as f:
            pickle.dump(times, f)
            f.close()
    if data_pkl_path:
        with open(data_pkl_path, 'wb') as f:
            pickle.dump(datas, f)
            f.close()
    if label_pkl_path:
        with open(label_pkl_path, 'wb') as f:
            pickle.dump(labels, f)
            f.close()
    logger.info('****** Times and Datas has been written into pkls! ******')

def single_locator(data: pd.DataFrame, name: str):
    """
    获取单个样本的 sample_i × dimensionality × sequence_length
    :param data:
    :param name:
    :return:
    """
    return data.loc[data['ID_M'].str.contains(name)].sort_values(by = 'B_checkweek_a_b')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) # 用于在终端输出 logging.info
    get_data_slice('./data/B-2022-02-23(孕14-40周检查）.csv', data_pkl_path = './temp/train_data_6.pkl',
                   time_pkl_path = './temp/train_time_6.pkl', label_pkl_path = './temp/train_label_6.pkl' ,number = 6)