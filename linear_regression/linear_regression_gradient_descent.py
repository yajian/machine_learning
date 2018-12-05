# coding=utf-8
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 10000  # 把样本集数据学习1000次
learning_rate = 0.01  # 步长 学习率 不能太大 太大容易来回震荡 太小 耗时间，跳不出局部最优解
# 可以写learn_rate动态变化，随着迭代次数越来越大 ，学习率越来越小 learning_rate/n_epoches
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 可以使用TensorFlow或者Numpy或者sklearn的StandardScaler去进行归一化
# 归一化可以最快的找到最优解
# 常用的归一化方式：
# 最大最小值归一化 (x-min)/(max-min)
# 方差归一化 x/方差
# 均值归一化 x-均值 结果有正有负 可以使调整时的速度越来越快。
scaler = StandardScaler().fit(housing_data_plus_bias)  # 创建一个归一化对象
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)  # 真正执行 因为来源于sklearn所以会直接执行，不会延迟。
housing_data_target = housing.target.reshape(-1, 1)

X = tf.placeholder(tf.float32, name='X')
y = tf.placeholder(tf.float32, name='y')

# random_uniform函数创建图里一个节点包含随机数值，给定它的形状和取值范围，就像numpy里面rand()函数
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')  # theta是参数 W0-Wn 一列 按照-1.0到1.0随机给
y_pred = tf.matmul(X, theta, name="predictions")  # 相乘 m行一列
error = y_pred - y  # 列向量和列向量相减 是一组数
mse = tf.reduce_mean(tf.square(error), name="mse")  # 误差平方加和，最小二乘 平方均值损失函数 手动实现
# 梯度的公式：(y_pred - y) * xj  i代表行 j代表列
gradients = 2.0 / m * tf.matmul(tf.transpose(X), error)  # 矩阵和向量相乘会得到新的向量 一组梯度
# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.assign(theta, theta - learning_rate * gradients)  # assigin赋值 算一组w
# training_op实际上就是需要迭代的公式

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 初始化

    for epoch in range(n_epochs):  # 迭代1000次
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ",
                  sess.run(mse, feed_dict={X: scaled_housing_data_plus_bias, y: housing_data_target}))  # 每运行100次的时候输出
        sess.run(training_op, feed_dict={X: scaled_housing_data_plus_bias, y: housing_data_target})

    best_theta = theta.eval()  # 最后的w参数值
    print(best_theta)
