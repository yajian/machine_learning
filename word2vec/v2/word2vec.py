# coding=utf-8

import os
import tensorflow as tf
from data import SkipGramDataSet
import numpy as np

dataset = SkipGramDataSet(os.path.join(os.path.curdir, 'test.txt'))

VOCAB_SIZE = dataset.vocab_size
print 'vocab_size:{}'.format(VOCAB_SIZE)
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.01

TRAIN_STEPS = 10000

BATCH_SIZE = 32
WINDOW_SIZE = 2


class Word2Vec(object):

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 输入层，维度为词集大小
            with tf.name_scope('inputs'):
                self.x = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)
                self.y = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)
            # 隐藏层，w1就是词向量
            with tf.name_scope('layer1'):
                self.W1 = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1), dtype=tf.float32,
                                      name='w1')
                self.b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE], dtype=tf.float32))
            # hidden是把输入的one-hot转化为词向量的结果
            hidden = tf.add(self.b1, tf.matmul(self.x, self.W1))
            with tf.name_scope('layer2'):
                self.W2 = tf.Variable(tf.random_uniform([EMBEDDING_SIZE, VOCAB_SIZE], -1, 1), dtype=tf.float32)
                self.b2 = tf.Variable(tf.random_normal([VOCAB_SIZE]), dtype=tf.float32)
            # 输出层是softmax求概率之后的结果
            self.prediction = tf.nn.softmax(tf.add(tf.matmul(hidden, self.W2), self.b2))
            # 损失函数是交叉熵
            log = self.y * tf.log(self.prediction)
            self.loss = tf.reduce_mean(-tf.reduce_sum(log, reduction_indices=[1], keep_dims=True))
            # 梯度下降
            self.opt = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)

    # 把词的下标值转化为one-hot表示
    def _one_hot_input(self, dataset):
        # features和labels记录了词在词集中的位置
        features, labels = dataset.generate_batch_inputs(BATCH_SIZE, WINDOW_SIZE)
        f, l = [], []
        for w in features:
            # 产生全0向量
            tmp = np.zeros([VOCAB_SIZE])
            # 下标位置置1
            tmp[w] = 1
            f.append(tmp)
        for w in labels:
            tmp = np.zeros([VOCAB_SIZE])
            tmp[w] = 1
            l.append(tmp)
        return f, l

    def train(self, dataset, n_iters, ):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_iters):
                features, labels = self._one_hot_input(dataset)
                predi, loss, w1 = sess.run([self.prediction, self.loss],
                                           feed_dict={self.x: features, self.y: labels})
                print 'loss:{}'.format(loss)


word2vec = Word2Vec()
word2vec.train(dataset, TRAIN_STEPS)
