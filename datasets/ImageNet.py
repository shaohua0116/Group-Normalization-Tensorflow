from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from skimage.io import imread
from scipy.misc import imresize
from datasets.imagenet.map import class2num

from util import log

__IMAGENET_IMG_PATH__ = '/YOUR_IMAGENET_PATH/ILSVRC/Data/CLS-LOC'
__IMAGENET_LIST_PATH__ = './datasets/imagenet'

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        file = os.path.join(__IMAGENET_IMG_PATH__, self._ids[0])

        try:
            imread(file)
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

    def load_image(self, id):
        img = imread(
            os.path.join(__IMAGENET_IMG_PATH__, id)) / 255. * 2 - 1
        img = imresize(img, [256, 256])

        y = np.random.randint(img.shape[0]-224)
        x = np.random.randint(img.shape[1]-224)
        img = img[y:y+224, x:x+224, :3]

        l = np.zeros(1000)
        l[class2num[id.split('/')[-2]]] = 1
        return img, l

    def get_data(self, id):
        # preprocessing and data augmentation
        m, l = self.load_image(id)
        return m, l

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __size__(self):
        return 114, 114

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(is_train=True, ratio=0.8):
    ids = all_ids()

    num_trains = int(len(ids) * ratio)

    dataset_train = Dataset(ids[:num_trains], name='train', is_train=False)
    dataset_test = Dataset(ids[num_trains:], name='test', is_train=False)
    return dataset_train, dataset_test


def all_ids():
    id_filename = 'train_list.txt'

    id_txt = os.path.join(__IMAGENET_LIST_PATH__, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    rs.shuffle(_ids)
    return _ids
