# coding=utf-8

import os
import tensorflow as tf
import collections
import numpy as np
import random


class DataSet(object):
    def __init__(self, file):
        self.file = file
        self.data_index = 0
        self._build_dataset()

    def _build_dataset(self):
        if not os.path.exists(self.file):
            raise ValueError("file doesn't exists --> {}".format(self.file))
        f = open(self.file, 'r')
        # 保存词集
        self.data = tf.compat.as_str(f.read()).split()
        if f:
            f.close()
        c = collections.Counter(self.data).most_common()
        # 计算词集大小
        self.vocab_size = len(c)
        self.counter = c.insert(0, ('UNK', -1))
        self.vocab_size += 1
        # 词-下标字典
        self.word2id = dict()
        # 下标-词字典
        self.id2word = dict()
        for word, _ in c:
            self.word2id[word] = len(self.word2id)
            self.id2word[len(self.id2word)] = word

    def generate_batch_inputs(self, batch_size, window_size):
        raise NotImplementedError()


class SkipGramDataSet(DataSet):
    def generate_batch_inputs(self, batch_size, window_size):
        features = np.ndarray(shape=(batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size,), dtype=np.int32)
        i = 0
        while True:
            if self.data_index == len(self.data):
                self.data_index = 0
            # 窗口的左侧位置
            left = max(0, self.data_index - window_size)
            # 窗口的右侧位置
            right = min(len(self.data), self.data_index + window_size + 1)
            # 遍历窗口里的每个单词
            for k in range(left, right):
                if k != self.data_index:
                    # 输入是中心词
                    features[i] = self.word2id[self.data[self.data_index]]
                    # label值是中心词周围在窗口内的值
                    labels[i] = self.word2id[self.data[k]]
                    i += 1
                    if i == batch_size:
                        return features, labels
            self.data_index += 1
