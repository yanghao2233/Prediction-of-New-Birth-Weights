# -*- coding:utf-8 -*-
# @Author: yanghao
# @Create time: 2022/02/07 11:36
# @Description: 主程序块

from sklearn.metrics import accuracy_score, roc_auc_score, auc

import pandas as pd
import numpy as np
from tqdm import trange, tqdm

import tensorflow as tf

import config

from tlstm import T_LSTM


def trainer(path:str, hidden_dim, fc_dim, model_path, key, learning_rate:float = 0.0001, epochs: int = 10, dropout_prob: float = 0.05, mode: str = 'T'):
    """
    利用神经网络进行训练的主函数。
    :param
        path: 训练数据储存母路径，格式为 string。
        learning_rate: 神经网络学习率，格式为 float。默认值为 0.0001。
        epochs: 神经网络迭代次数，格式为 int。默认值为 10。
        dropout_prob: 神经网络 drop 率， 格式为 float。默认值为0.05。
        hidden_dim: 隐藏层的维度，格式为 int。
        fc_dim: fc 维度，格式为 int。
        model_path: 模型储存路径，格式为 string。
        mode: 训练模式，格式为 str。默认值为 ‘T’，可选值为 ‘T’， ‘VTL“， ’C‘。
    :return:
    """

    path_train = path + './temp/train_data_6.PKL'
    data_train_batches = pd.read_pickle(path_train)

    path_time_elapse = './temp/train_time_6.pkl'
    elapse_train_batches = pd.read_pickle(path_time_elapse)

    path_labels = path + './temp/train_label_6.pkl'
    label_train_batches = pd.read_pickle(path_labels)

    num_train_batches = data_train_batches.shape[1]-1

    input_dim = data_train_batches[0].shape[1]
    output_dim = data_train_batches[0].shape[1]
    print(num_train_batches, input_dim, output_dim)

    TLSTM = T_LSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    cross_entropy, y_pred, y, logits, labels = TLSTM.get_cost_acc()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in trange(epochs):

            for i in range(num_train_batches):
                batch_xs, batch_ys, batch_ts = data_train_batches[i], label_train_batches[i], elapse_train_batches[i]
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
                sess.run(optimizer, feed_dict = {TLSTM.input: batch_xs, TLSTM.labels: batch_ys,
                                                 TLSTM.keep_prob: dropout_prob, TLSTM.time: batch_ts})

        print('Train Session has been done!')
        saver.save(sess, model_path)
        print('Model has been saved')

        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []
        for i in range(num_train_batches):
            batch_xs, batch_ys, batch_ts = data_train_batches[i], label_train_batches[i], elapse_train_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(TLSTM.get_cost_acc(), feed_dict={
                TLSTM.input:
                    batch_xs, TLSTM.labels: batch_ys, TLSTM.keep_prob: dropout_prob, TLSTM.time: batch_ts})

            if i > 0:
                Y_true = np.concatenate([Y_true, y_train], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
                Labels = np.concatenate([Labels, labels_train], 0)
                Logits = np.concatenate([Logits, logits_train], 0)
            else:
                Y_true = y_train
                Y_pred = y_pred_train
                Labels = labels_train
                Logits = logits_train

        total_acc = accuracy_score(Y_true, Y_pred)
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
        print("Train Accuracy = {:.3f}".format(total_acc))
        print("Train AUC = {:.3f}".format(total_auc))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))

def tester():
    pass

def main():
    # if config.mode == 0:
    trainer(path = config.train_data_path, hidden_dim = config.hidden_dim, fc_dim = config.fc_dim, model_path = config.model_path, key = config.if_init_weight,
             learning_rate = config.learning_rate, epochs = config.train_epochs, dropout_prob = config.drop_prob, mode = config.train_mode)
    # if config.mode == 1:
    #     tester()

if __name__ == '__main__':
    main()