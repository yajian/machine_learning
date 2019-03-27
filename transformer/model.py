# coding=utf-8
import load_data
from modules import get_token_embeddings, positional_encoding, multihead_attention, ff, label_smoothing, noam_scheme
import tensorflow as tf
from load_data import get_batch
from hparams import Hyperparams as hp


class Transformer(object):
    def __init__(self, training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if training:
                self.x, self.y, self.num_batch = get_batch()
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)

            de2idx, idx2de = load_data.load_vocab('./preprocessed/de.vocab.tsv')
            en2idx, idx2en = load_data.load_vocab('./preprocessed/en.vocab.tsv')

            self.embedding = get_token_embeddings(len(de2idx), hp.hidden_units, zero_pad=True)

            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                self.enc = tf.nn.embedding_lookup(self.embedding, self.x)
                # scale
                self.enc *= hp.hidden_units ** 0.5
                # positional encoding
                self.enc += positional_encoding(self.enc)
                self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=training)
                for i in range(hp.num_blocks):
                    with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       values=self.enc,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=training,
                                                       causality=False)
                        self.enc = ff(self.enc, num_units=[hp.d_ff, hp.hidden_units])

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                self.dec = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
                self.dec *= hp.hidden_units ** 0.5
                self.dec += positional_encoding(self.dec)
                self.dec = tf.layers.dropout(self.dec, hp.dropout_rate, training=training)
                for i in range(hp.num_blocks):
                    with tf.variable_scope('num_block_{}'.format(i), reuse=tf.AUTO_REUSE):
                        self.dec = multihead_attention(queries=self.dec, keys=self.dec, values=self.dec,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=training,
                                                       causality=True,
                                                       scope='self_attention')
                        self.dec = multihead_attention(
                            queries=self.dec,
                            keys=self.enc,
                            values=self.enc,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            causality=False,
                            scope='vanilla_attention'
                        )
                        self.dec = ff(self.dec, num_units=[hp.d_ff, hp.hidden_units])
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
            if training:
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / tf.reduce_sum(self.istarget)

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
