# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 数据集
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
# 超参数
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1
# 输入数据
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax函数
pred = tf.nn.softmax(tf.matmul(x, W))
# 损失函数，这里因为y用one_hot表示，所以可以直接用矩阵代替指示函数I
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# W梯度下降
W_grad = -tf.matmul(tf.transpose(x), y - pred)
# b梯度下降
b_grad = -tf.reduce_mean(-tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)
# W更新方式
new_W = W.assign(W - learning_rate * W_grad)
# b更新方式
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})
