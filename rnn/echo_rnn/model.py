# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 100
# 共50000个数据
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
# 回声滞后输入3个时间单位
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


# 生成数据
def generateData():
    # 0，1采样，产生50000个数据
    # 选0和选1的概率都是0.5
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    # 把x循环位移3位
    y = np.roll(x, echo_step)
    # 前面的几项置0
    y[0:echo_step] = 0
    # x的形状（5，10000）
    x = x.reshape((batch_size, -1))
    # y的形状（5，10000）
    y = y.reshape((batch_size, -1))
    return (x, y)


# 构造输入序列
# x的形状（5，15）
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
# y的形状（5，15）
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
# 按列解包，可以理解为转置，inputs_series目前是一个长度为15的list，list的每一项是一个长度为5的array
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# 形状是（5，4）
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

current_state = init_state
state_series = []
for current_input in inputs_series:
    # 对inputs_series的每一项进行reshape，current_input是长度为5的array，reshape之后变成二维矩阵形状为（5，1）
    current_input = tf.reshape(current_input, [batch_size, 1])
    # 把当前输入项和当前状态合并，axis=1表示列，current_state：（5，4），current_input（5，1），合并之后是（5，5）
    # 这里实现了St=f(U*Xt+W*St−1)，把U、W合并，Xt、St-1合并
    input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)
    # 矩阵乘法（5，5）* （5，4）=（5，4）和之前的state结构一样
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    # 把新的state放入数组
    state_series.append(next_state)
    current_state = next_state
# state:(5,4) W2:(4,2)，最终得到（5，2）
logits_series = [tf.matmul(state, W2) + b2 for state in state_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = []
for logits, labels in zip(logits_series, labels_series):
    losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    for epoch_idx in range(num_epochs):
        # 产生数据
        x, y = generateData()
        _current_state = np.zeros((batch_size, state_size))
        print 'New data,epoch ', epoch_idx
        for batch_idx in range(num_batches):
            # 保持5行不变，每次截取truncated_backprop_length长度的数据
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={batchX_placeholder: batchX,
                           batchY_placeholder: batchY,
                           init_state: _current_state})
            loss_list.append(_total_loss)
            if batch_idx % 100 == 0:
                print 'step ', batch_idx, 'loss ', _total_loss
                plot(loss_list, _predictions_series, batchX, batchY)
plt.ioff()
plt.show()
