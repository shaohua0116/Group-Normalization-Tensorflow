from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from util import log
from pprint import pprint

from model import Model
from input_ops import create_input_ops

import os
import time
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf


class Trainer(object):
    def __init__(self,
                 config,
                 dataset,
                 dataset_test):
        self.config = config
        hyper_parameter_str = '{}_lr_{}_bs_{}_norm_type_{}'.format(
            config.dataset, config.learning_rate,
            config.batch_size, config.norm_type
        )
        self.train_dir = './train_dir/%s-%s-%s' % (
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(dataset, self.batch_size,
                                               is_training=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)

        # --- create model ---
        self.model = Model(config)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate

        self.check_op = tf.no_op()

        all_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer_loss'
        )

        self.train_summary_op = tf.summary.merge_all(key='train')
        self.test_summary_op = tf.summary.merge_all(key='test')

        self.saver = tf.train.Saver(max_to_keep=100)
        self.pretrain_saver = tf.train.Saver(var_list=tf.trainable_variables(),
                                             max_to_keep=100)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.log_step = self.config.log_step
        self.test_sample_step = self.config.test_sample_step
        self.write_summary_step = self.config.write_summary_step

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.pretrain_saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)

        max_steps = 1000000

        ckpt_save_step = self.config.ckpt_save_step
        log_step = self.log_step
        test_sample_step = self.test_sample_step
        write_summary_step = self.write_summary_step

        for s in xrange(max_steps):
            # periodic inference
            if s % test_sample_step == 0:
                step, accuracy, test_summary, loss, step_time = \
                    self.run_test(self.batch_test, is_train=False)
                self.log_step_message(step, accuracy, loss, step_time, is_train=False)
                self.summary_writer.add_summary(test_summary, global_step=step)

            step, accuracy, train_summary, loss, step_time = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % log_step == 0:
                self.log_step_message(step, accuracy, loss, step_time)

            if s % write_summary_step == 0:
                self.summary_writer.add_summary(train_summary, global_step=step)

            if s % ckpt_save_step == 0 and s > 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(self.session,
                                os.path.join(self.train_dir, 'model'),
                                global_step=step)

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.train_summary_op,
                 self.model.loss, self.check_op, self.optimizer]

        fetch_values = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, accuracy, summary, loss] = fetch_values[:4]

        _end_time = time.time()

        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def run_test(self, batch, is_train=False, repeat_times=8):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        step, accuracy, summary, loss = self.session.run(
            [self.global_step, self.model.accuracy,
             self.test_summary_op, self.model.loss],
            feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False)
        )

        _end_time = time.time()

        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def log_step_message(self, step, accuracy, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "Accuracy: {accuracy:.2f}% "
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         accuracy=accuracy*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'Fashion', 'SVHN', 'CIFAR10'])
    parser.add_argument('--norm_type', type=str, default='batch',
                        choices=['batch', 'group'])
    # Log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--test_sample_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=10)
    parser.add_argument('--ckpt_save_step', type=int, default=1000)
    # Learning
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    config = parser.parse_args()

    if config.dataset == 'MNIST':
        import datasets.mnist as dataset
    elif config.dataset == 'Fashion':
        import datasets.fashion_mnist as dataset
    elif config.dataset == 'SVHN':
        import datasets.svhn as dataset
    elif config.dataset == 'CIFAR10':
        import datasets.cifar10 as dataset
    else:
        raise ValueError(config.dataset)

    dataset_train, dataset_test = dataset.create_default_splits()
    image, label = dataset_train.get_data(dataset_train.ids[0])
    config.data_info = np.concatenate([np.asarray(image.shape), np.asarray(label.shape)])

    trainer = Trainer(config,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f", config.dataset, config.learning_rate)
    trainer.train()

if __name__ == '__main__':
    main()
