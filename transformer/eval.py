# coding=utf-8

from model import Transformer
from load_data import load_test_data, load_vocab
import tensorflow as tf
from hparams import Hyperparams as hp
import os
import codecs
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def eval():
    transformer = Transformer(training=False)

    X, Sources, Targets = load_test_data()
    en2idx, idx2en = load_vocab('./preprocessed/en.vocab.tsv')

    with transformer.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print 'restored'
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name
            if not os.path.exists('results'):
                os.makedirs('results')
            with codecs.open('results/' + mname, 'w', 'utf-8') as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.max_len), np.int32)
                    for j in range(hp.max_len):
                        _preds = sess.run(transformer.preds, {transformer.x: x, transformer.y: preds})
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write('- source: {}\n'.format(source))
                        fout.write('- expected: {}\n'.format(target))
                        fout.write('- got: {}\n\n'.format(got))

                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append(ref)
                            hypotheses.append(hypothesis)
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()
    print 'Done'
