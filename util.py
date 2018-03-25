""" Utilities """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Logging
# =======

import logging
from colorlog import ColoredFormatter
import tensorflow as tf


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('Log')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov


def train_test_summary(name, value, max_outputs=4, summary_type='scalar'):
    if summary_type == 'scalar':
        tf.summary.scalar(name, value, collections=['train'])
        tf.summary.scalar("test_{}".format(name), value, collections=['test'])
    elif summary_type == 'image':
        tf.summary.image(name, value, max_outputs=max_outputs, collections=['train'])
        tf.summary.image("test_{}".format(name), value,
                         max_outputs=max_outputs, collections=['test'])
