# coding=utf-8

import codecs
from hparams import Hyperparams as hp
import regex
import tensorflow as tf
import numpy as np


def load_vocab(path):
    with codecs.open(path, 'r') as file:
        lines = file.readlines()
        vocabs = []
        for line in lines:
            items = line.split('\t')
            if int(items[1]) > hp.min_cnt:
                vocabs.append(items[0])
        word2idx = {word: idx for idx, word in enumerate(vocabs)}
        idx2word = {idx: word for idx, word in enumerate(vocabs)}
        return word2idx, idx2word


def create_data(source_sents, target_sents):
    de2idx, idx2de = load_vocab('./preprocessed/de.vocab.tsv')
    en2idx, idx2en = load_vocab('./preprocessed/en.vocab.tsv')
    x_list = []
    y_list = []
    Sources = []
    Targets = []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= hp.max_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    X = np.zeros([len(x_list), hp.max_len], np.int32)
    Y = np.zeros([len(y_list), hp.max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.max_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.max_len - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(hp.source_train, 'r', 'utf-8').read().split('\n') if line and line[0] != '<']
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(hp.target_train, 'r', 'utf-8').read().split('\n') if line and line[0] != '<']
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y


def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets  # (1064, 150)


def get_batch():
    X, Y = load_train_data()

    num_batch = len(X) // hp.batch_size
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    input_queues = tf.train.slice_input_producer([X, Y])

    x, y = tf.train.shuffle_batch(
        input_queues,
        num_threads=8,
        batch_size=hp.batch_size,
        capacity=hp.batch_size * 64,
        min_after_dequeue=hp.batch_size * 32,
        allow_smaller_final_batch=False
    )
    return x, y, num_batch


if __name__ == '__main__':
    # word2idx, idx2word = load_vocab(
    #     '/Users/huangyajian/demo/blog_project/machine_learning/transformer/preprocessed/de.vocab.tsv')
    # print word2idx
    # print idx2word

    x, y, num_batch = get_batch()
    print num_batch
