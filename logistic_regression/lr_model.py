# coding=utf-8
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
data_x = np.array(iris['data'])
data_y = np.array(iris['target']).reshape(-1, 1)
print data_x.shape
print data_y.shape
enc = OneHotEncoder()
enc.fit(data_y)
# 多分类问题，对label进行one-hot编码
targets = enc.transform(data_y).toarray()
# 输入行数不指定，列数为特征个数
X = tf.placeholder(dtype=tf.float32, shape=(None, data_x.shape[1]))
# 输出行数不指定，列数类别个数
y = tf.placeholder(dtype=tf.float32, shape=(None, 3))
# 学习率
alpha = 0.0001
# 迭代次数
epoch = 500
# 权重
theta = tf.Variable(tf.random_uniform([data_x.shape[1], 3], -1.0, 1.0), name='theta')
# 偏置
b = tf.Variable(tf.random_uniform([3], -1.0, 1.0), name='b')
# 预测值
predict_y = tf.nn.softmax(tf.matmul(X, theta) + b)
# 误差
error = tf.cast(y, tf.float32) - predict_y
# 代价函数，使用交叉熵函数，手动实现
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y), reduction_indices=1))
# 权重的负梯度
theta_gradient = -tf.matmul(tf.matrix_transpose(X), error, name='theta_gradient')
# 偏置的负梯度
b_gradient = - tf.reduce_mean(tf.matmul(tf.transpose(X), error), reduction_indices=0, name='b_gradient')
# 权重迭代项
training_op1 = tf.assign(theta, theta - alpha * theta_gradient)
# 偏置迭代项
training_op2 = tf.assign(b, b - alpha * b_gradient)
# 全局初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(epoch):
        sess.run([training_op1, training_op2, cost], feed_dict={X: data_x, y: targets})
        print sess.run(theta, feed_dict={X: data_x, y: targets})
        print sess.run(b, feed_dict={X: data_x, y: targets})
        print sess.run(cost, feed_dict={X: data_x, y: targets})
