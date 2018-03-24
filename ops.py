import tensorflow as tf
import tensorflow.contrib.slim as slim
from util import log


def norm(x, norm_type, is_train, G=32, esp=1e-5):
    log.infov('Normalization: {}'.format(norm_type))
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [N, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [N, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, info=False,
           activation_fn=lrelu, norm_type='batch',
           k=4, s=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s, s, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        activation = activation_fn(conv + biases)
        output = norm(activation, norm_type, is_train)
    if info: log.info('{} {}'.format(name, output))
    return output


def fc(input, output_shape, is_train, info=False,
       norm_type='batch', activation_fn=lrelu, name="fc"):
    activation = slim.fully_connected(input, output_shape, activation_fn=activation_fn)
    output = norm(activation, norm_type, is_train)
    if info: log.info('{} {}'.format(name, output))
    return output
