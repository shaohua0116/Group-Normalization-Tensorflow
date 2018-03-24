from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import conv2d, fc
from util import log, train_test_summary


class Model(object):

    def __init__(self, config, debug_information=False, is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.num_class = self.config.data_info[3]
        self.norm_type = self.config.norm_type

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'],  # [B, h, w, c]
            self.label: batch_chunk['label'],  # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.num_class

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        # Classifier: takes images as input and tries to output class label [B, n]
        def C(img, scope='Classifier', reuse=False):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                _ = img

                # conv layers
                num_channels = [64, 128, 256, 512]
                for i in range(len(num_channels)):
                    _ = conv2d(_, num_channels[i], is_train, norm_type=self.norm_type,
                               info=not reuse, name='conv_{}'.format(i))
                    _ = slim.dropout(_, keep_prob=0.5, is_training=is_train)

                # fc layers
                _ = tf.reshape(_, [self.batch_size, -1])
                num_fc_channels = [512, 128, 32, n]
                for i in range(len(num_fc_channels)):
                    _ = fc(_, num_fc_channels[i], is_train, norm_type='none',
                           info=not reuse, name='fc_{}'.format(i))
                return _

        logits = C(self.image)
        self.entropy, self.accuracy = build_loss(logits, self.label)
        self.loss = self.entropy

        train_test_summary("loss/accuracy", self.accuracy)
        train_test_summary("loss/loss", self.loss)
        train_test_summary("img/image", self.image, summary_type='image')
        log.warn('Successfully loaded the model.')
