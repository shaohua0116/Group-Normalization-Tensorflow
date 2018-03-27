from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from skimage.io import imread
from scipy.misc import imresize

from util import log

__IMAGENET_IMG_PATH__ = './datasets/tiny_imagenet/tiny-imagenet-200/'
__IMAGENET_LIST_PATH__ = './datasets/tiny_imagenet'

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

        with open(os.path.join(__IMAGENET_IMG_PATH__, 'wnids.txt')) as f:
            self.label_list = f.readlines()
        self.label_list = [label.strip() for label in self.label_list]

        with open(os.path.join(__IMAGENET_IMG_PATH__, 'val/val_annotations.txt')) as f:
            self.val_label_list = f.readlines()
        self.val_label_list = [label.split('\t')[1] for label in self.val_label_list]
        try:
            imread(file)
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

    def load_image(self, id):
        img = imread(
            os.path.join(__IMAGENET_IMG_PATH__, id))/255.
        img = imresize(img, [72, 72])
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        y = np.random.randint(img.shape[0]-64)
        x = np.random.randint(img.shape[1]-64)
        img = img[y:y+64, x:x+64, :3]

        l = np.zeros(200)
        if id.split('/')[1] == 'train':
            l[self.label_list.index(id.split('/')[-3])] = 1
        elif id.split('/')[1] == 'val':
            img_idx = int(id.split('/')[-1].split('_')[-1].split('.')[0])
            l[self.label_list.index(self.val_label_list[img_idx])] = 1
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
        return 64, 64

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(is_train=True, ratio=0.8):
    id_train, id_test = all_ids()

    dataset_train = Dataset(id_train, name='train', is_train=False)
    dataset_test = Dataset(id_test, name='test', is_train=False)
    return dataset_train, dataset_test


def all_ids():
    id_train_path = os.path.join(__IMAGENET_LIST_PATH__, 'train_list.txt')
    id_val_path = os.path.join(__IMAGENET_LIST_PATH__, 'val_list.txt')
    try:
        with open(id_train_path, 'r') as fp:
            id_train = [s.strip() for s in fp.readlines() if s]
        with open(id_val_path, 'r') as fp:
            id_val = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    rs.shuffle(id_train)
    rs.shuffle(id_val)
    return id_train, id_val
