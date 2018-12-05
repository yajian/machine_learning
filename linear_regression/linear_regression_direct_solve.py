# coding=utf-8
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

n_epochs = 10000  # 把样本集数据学习1000次
learning_rate = 0.01  # 步长 学习率 不能太大 太大容易来回震荡 太小 耗时间，跳不出局部最优解
# 可以写learn_rate动态变化，随着迭代次数越来越大 ，学习率越来越小 learning_rate/n_epoches
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
housing_data_target = housing.target.reshape(-1, 1)
X = tf.constant(housing_data_plus_bias, tf.float32)
y = tf.constant(housing_data_target, tf.float32)
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()
    print theta_value
