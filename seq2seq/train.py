# coding=utf-8

import random
import helper
from model import Seq2seq
import time
import tensorflow as tf


class config(object):
    embedding_dim = 100
    hidden_dim = 50
    batch_size = 128
    learning_rate = 0.005
    source_vocab_size = None
    target_vocab_size = None


def get_batch(doc_source, w2i_source, doc_target, w2i_target, batch_size):
    ps = []
    while len(ps) < batch_size:
        ps.append(random.randint(0, len(doc_source) - 1))
    source_batch = []
    target_batch = []

    source_lens = [len(doc_source[p]) for p in ps]
    target_lens = [len(doc_target[p]) + 1 for p in ps]

    max_source_len = max(source_lens)
    max_target_len = max(target_lens)

    for p in ps:
        source_seq = [w2i_source[w] for w in doc_source[p].split()] + [w2i_source["<PAD>"]] * (
                max_source_len - len(doc_source[p].split()))
        target_seq = [w2i_target[w] for w in doc_target[p].split()] + [w2i_target["<PAD>"]] * (
                max_target_len - 1 - len(doc_target[p].split())) + [w2i_target["<EOS>"]]
        source_batch.append(source_seq)
        target_batch.append(target_seq)
    return source_batch, source_lens, target_batch, target_lens


if __name__ == '__main__':
    print 'loading data ...'
    doc_source = helper.load_file('./data/small_vocab_en.txt')
    doc_target = helper.load_file('./data/small_vocab_fr.txt')
    s_token2idx, s_idx2token = helper.load_vocab('./data/small_vocab_en.txt', helper.SOURCE_CODES)
    t_token2idx, t_idx2token = helper.load_vocab('./data/small_vocab_fr.txt', helper.TARGET_CODES)
    print 'building model...'
    config = config()
    config.source_vocab_size = len(s_token2idx)
    config.target_vocab_size = len(t_token2idx)
    model = Seq2seq(config, t_token2idx, useTeacherForcing=True)
    batches = 10000
    print_every = 100
    print 'run model...'
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        losses = []
        total_loss = 0
        for batch in range(batches):
            source_batch, source_lens, target_batch, target_lens = get_batch(doc_source, s_token2idx, doc_target,
                                                                             t_token2idx, config.batch_size)

            feed_dict = {
                model.seq_inputs: source_batch,
                model.seq_inputs_len: source_lens,
                model.seq_targets: target_batch,
                model.seq_targets_len: target_lens}

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            total_loss += loss
            if batch % print_every == 0 and batch > 0:
                print_loss = total_loss if batch == 0 else total_loss / print_every
                losses.append(print_loss)
                total_loss = 0
                print("-----------------------------")
                print("batch:", batch, "/", batches)
                print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("loss:", print_loss)

                print("samples:\n")
                predict_batch = sess.run(model.out, feed_dict)
                for i in range(3):
                    print("in:", [s_idx2token[num] for num in source_batch[i] if s_idx2token[num] != "<PAD>"])
                    print("out:", [t_idx2token[num] for num in predict_batch[i] if t_idx2token[num] != "<PAD>"])
                    print("tar:", [t_idx2token[num] for num in target_batch[i] if t_idx2token[num] != "<PAD>"])
                    print("")
        print losses
        print saver.save(sess, "checkpoint/model.ckpt")
