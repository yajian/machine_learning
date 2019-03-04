# coding = utf-8
import tensorflow as tf


class Seq2seq(object):

    def build_inputs(self, config):
        self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_len = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_len = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')

    def build_loss(self, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.seq_targets, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss

    def build_optim(self, loss, lr):
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    def __init__(self, config, word2int_target, useTeacherForcing=True):
        self.build_inputs(config)

        with tf.variable_scope('encoder'):
            encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedding = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
            with tf.variable_scope('gru_cell'):
                encoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedding,
                                                                sequence_length=self.seq_inputs_len,
                                                                dtype=tf.float32)

        with tf.variable_scope('decoder'):
            decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='decoder_embedding')
            with tf.variable_scope('gru_cell'):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                decoder_initial_state = encoder_states

            tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_go') * word2int_target['<GO>']
            tokens_eos = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_eos') * word2int_target['<EOS>']
            tokens_eos_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_eos)
            tokens_go_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_go)

            W = tf.Variable(tf.random_uniform([config.hidden_dim, config.target_vocab_size]), dtype=tf.float32,
                            name='decoder_out_w')
            b = tf.Variable(tf.random_uniform([config.target_vocab_size]), dtype=tf.float32, name='decoder_out_b')

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:
                    initial_elements_finished = (0 >= self.seq_targets_len)
                    initial_state = decoder_initial_state
                    initial_input = tokens_go_embedded
                    initial_output = None
                    initial_loop_state = None
                    return (initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state)
                else:
                    def get_next_input():
                        if useTeacherForcing:
                            prediction = self.seq_targets[:, time - 1]
                        else:
                            output_logits = tf.add(tf.matmul(previous_output, W), b)
                            prediction = tf.argmax(output_logits, axis=1)
                        next_input = tf.nn.embedding_lookup(decoder_embedding, prediction)
                        return next_input

                    elements_finished = (time >= self.seq_targets_len)
                    finished = tf.reduce_all(elements_finished)
                    input = tf.cond(finished, lambda: tokens_eos_embedded, lambda: get_next_input())
                    state = previous_state
                    output = previous_output
                    loop_state = None
                    return (elements_finished, input, state, output, loop_state)

            decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()
            decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2])

            decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim))
            decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
            decoder_logits = tf.reshape(decoder_logits_flat,
                                        (decoder_batch_size, decoder_max_steps, config.target_vocab_size))
        self.out = tf.argmax(decoder_logits, 2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.seq_targets, logits=decoder_logits)
        sequence_mask = tf.sequence_mask(self.seq_targets_len, dtype=tf.float32)
        loss = loss * sequence_mask
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
