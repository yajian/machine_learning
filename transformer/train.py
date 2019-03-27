# coding=utf-8

import tensorflow as tf
from tqdm import tqdm

from hparams import Hyperparams as hp
from model import Transformer

transformer = Transformer()
sv = tf.train.Supervisor(graph=transformer.graph, logdir=hp.logdir, save_model_secs=0)
with sv.managed_session() as sess:
    for epoch in range(1, hp.num_epochs):
        if sv.should_stop():
            break
        for step in tqdm(range(transformer.num_batch), total=transformer.num_batch, ncols=70, leave=False, unit='b'):
            sess.run(transformer.train_op)
            loss, acc = sess.run([transformer.mean_loss, transformer.acc])
            print('loss:{},acc:{}'.format(loss, acc))
        gs = sess.run(transformer.global_step)
        sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    print 'Done'
