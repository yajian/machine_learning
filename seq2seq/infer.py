# coding=utf-8
import tensorflow as tf
import numpy as np
import random
import time
from model import Seq2seq
import helper
from train import config, get_batch

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

model_path = "checkpoint/model.ckpt"

if __name__ == "__main__":
    print("(1)load data......")
    docs_source = ['new jersey is usually hot during autumn , and it is never quiet in winter .\n']
    docs_target = ["new jersey est généralement chaud pendant l' automne , et il est jamais calme en hiver .\n"]
    w2i_source, i2w_source = helper.load_vocab('./data/small_vocab_en.txt', helper.SOURCE_CODES)
    w2i_target, i2w_target = helper.load_vocab('./data/small_vocab_fr.txt', helper.TARGET_CODES)

    print("(2) build model......")
    config = config()
    config.source_vocab_size = len(w2i_source)
    config.target_vocab_size = len(w2i_target)
    model = Seq2seq(config, w2i_target, useTeacherForcing=False)

    print("(3) run model......")
    print_every = 100
    max_target_len = 20

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target,
                                                                         w2i_target, config.batch_size)

        feed_dict = {
            model.seq_inputs: source_batch,
            model.seq_inputs_len: source_lens,
            model.seq_targets: [[0] * max_target_len] * len(source_batch),
            model.seq_targets_len: [max_target_len] * len(source_batch)
        }

        print("samples:\n")
        predict_batch = sess.run(model.out, feed_dict)
        for i in range(3):
            print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "<PAD>"])
            print("out:", [i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "<PAD>"])
            print("tar:", [i2w_target[num] for num in target_batch[i] if i2w_target[num] != "<PAD>"])
            print("")
