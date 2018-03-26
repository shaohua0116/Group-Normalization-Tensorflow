from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import h5py

from util import log

__PATH__ = './datasets/cifar10'

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = 'data.hdf5'

        file = os.path.join(__PATH__, filename)
        log.info("Reading %s ...", file)

        try:
            self.data = h5py.File(file, 'r')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

    def get_data(self, id):
        # preprocessing and data augmentation
        m = self.data[id]['image'].value/255.
        l = self.data[id]['label'].value.astype(np.float32)
        return m, l

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(is_train=True):
    id_train, id_test = all_ids(50000)

    dataset_train = Dataset(id_train, name='train', is_train=False)
    dataset_test = Dataset(id_test, name='test', is_train=False)
    return dataset_train, dataset_test


def all_ids(num_trains):
    id_filename = 'id.txt'

    id_txt = os.path.join(__PATH__, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

    id_train, id_test = _ids[:num_trains], _ids[num_trains:]
    rs.shuffle(id_train)
    rs.shuffle(id_test)

    return id_train, id_test
