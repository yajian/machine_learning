# coding=utf-8
# 参考https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print 'found and verified {}'.format(filename)
    else:
        print statinfo.st_size
        raise Exception('Failed to verify {}. Can you get to it with a browser?'.format(filename))
    return filename


filename = maybe_download('./text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data('./text8.zip')
print 'data size {}'.format(len(words))

vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 取出前5w个词汇，key是word，values是按频次排序下标
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 存储所有单词频次排序，排序超过5w置0
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print 'most common words (+UNK) {}'.format(count[:5])
print 'Sample data {} {}'.format(data[: 10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # 这里是说：样本个数batch_size=每个单词采样数量num_skips*单词数量（即后面的i）
    assert batch_size % num_skips == 0
    # 以目标单词为中心，skip_window为半径，所以每个单词产生样本个数最多为2 * skip_window
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.float32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    span = 2 * skip_window + 1
    # buffer相当于一个滑动窗口，每次往右移动1个单位，存储了目标单词周围+-skip_window范围内的单词
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            # 用过一次的词汇记录下来，避免重复采样
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        # 移动滑动窗口
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=128, num_skips=2, skip_window=2)
for i in range(8):
    print batch[i], reverse_dictionary[batch[i]], labels[i, 0], reverse_dictionary[labels[i, 0]]

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        # 这里初始化所有词汇的向量表达形式，对于这个例子是5000个单词的128维表示
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # train_inputs只是记录了输入向量的下标，这里取出真正的向量输入
        embed = tf.nn.embedding_lookup(embedding, train_inputs)
        # 权重
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

num_step = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print 'initialized'
    avg_loss = 0
    for step in range(num_step):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        avg_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            print 'Average loss at step {}: {}'.format(step, avg_loss)
            avg_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {}: '.format(valid_word)
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '{}{},'.format(log_str, close_word)
                print log_str
            final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embedding'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
