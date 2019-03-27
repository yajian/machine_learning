# coding=utf-8
import tensorflow as tf
import numpy as np


def get_token_embeddings(vocab_size, hidden_unit, zero_pad=True):
    with tf.variable_scope('shared_weight_matrix'):
        embeddings = tf.get_variable('weight_mat', dtype=tf.float32, shape=(vocab_size, hidden_unit),
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, hidden_unit]), embeddings[1:, :]), 0)
        return embeddings


def positional_encoding(inputs, masking=True, scope='positional_encoding'):
    # N代表batch_size,T代表max_len,
    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]
    E = inputs.get_shape().as_list()[2]
    print 'N:{},T:{},E:{}'.format(N, T, E)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # expand_dims函数产生一条样本每个位置的id，tile函数用于扩增到N条样本
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # 位置编码第一部分，这里使用max_len是因为
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / E) for i in range(E)]
            for pos in range(T)])
        # 位置编码第二部分
        # dim 2i
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # dim 2i+1
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        return tf.to_float(outputs)


def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0., training=True,
                                 scope='scaled_dot_product_attention'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        # 这里对K进行转置，第0维是batch所以不用转置
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        # scale
        outputs /= d_k ** 0.5
        # key masking
        outputs = mask(outputs, Q, K, type='key')

        if causality:
            outputs = mask(outputs, type='future')
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image('attention', tf.expand_dims(attention[:1], -1))
        outputs = mask(outputs, Q, K, type='query')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)
    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ('k', 'key', 'keys'):
        # 经过这一步，矩阵形状变为(N,T_k)
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    elif type in ('q', 'query', 'queries'):
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(queries)[1]])
        outputs = inputs * masks
    elif type in ('f', 'future', 'right'):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print ' check if you entered type correctly!'
    return outputs


def layer_normalization(inputs, epsilon=1e-8, scope='layer_normalization'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.zeros_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta
        return outputs


def multihead_attention(queries, keys, values, num_heads=8, dropout_rate=0, training=True, causality=False,
                        scope='multihead_attention'):
    # embedding size
    d_model = queries.get_shape()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model)
        K = tf.layers.dense(keys, d_model)
        V = tf.layers.dense(values, d_model)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = layer_normalization(outputs)
    return outputs


def ff(inputs, num_units, scope='positionwise_feedforward'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = layer_normalization(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / K)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
