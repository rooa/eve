"""datasets.py: wrappers around Keras datasets."""

import tarfile
from abc import ABCMeta
from functools import reduce

import numpy as np
from keras.datasets import cifar10, cifar100, imdb
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from scipy.misc import imresize
from tqdm import tqdm

from eve.exp.utils import (get_stories, vectorize_stories, get_ptb,
                           ptb_raw_data, QFILE)

class Dataset(metaclass=ABCMeta):

    """Abstract dataset loader+preprocessor."""

    def __init__(self, loader, max_samples):
        """Load a dataset.

        Arguments:
            loader: callable: returns (X, y), (X, y)
            max_samples: int or None: if not none, keep at most this many
                samples.
        """
        (self.X_train, self.y_train), (self.X_test, self.y_test) = loader()
        if max_samples is not None:
            self.X_train = self.X_train[:max_samples]
            self.y_train = self.y_train[:max_samples]
            self.X_test = self.X_test[:max_samples]
            self.y_test = self.y_test[:max_samples]

    @property
    def train_data(self):
        """X_train, y_train."""
        return self.X_train, self.y_train

    @property
    def test_data(self):
        """X_test, y_test."""
        return self.X_test, self.y_test

    @property
    def input_shape(self):
        """Shape of X excluding the number of samples."""
        return self.X_train.shape[1:]


class Cifar10(Dataset):

    """Cifar10 dataset with simple pixel rescaling for preprocessing."""

    def __init__(self, max_samples=None):
        super().__init__(cifar10.load_data, max_samples)
        self.X_train = self.X_train.astype(K.floatx()) / 255.
        self.X_test = self.X_test.astype(K.floatx()) / 255.
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    @property
    def num_classes(self):
        """Number of output labels."""
        return 10


class Cifar100(Dataset):

    """Cifar100 dataset processed same as Cifar10."""

    def __init__(self, max_samples=None, resize=None):
        super().__init__(cifar100.load_data, max_samples)

        if resize is not None:
            if K.image_data_format() == "channels_first":
                # Bring channels to the end for scipy
                self.X_train = np.transpose(self.X_train, (0, 2, 3, 1))
                self.X_test = np.transpose(self.X_test, (0, 2, 3, 1))

            self.X_train = np.stack([imresize(i, resize) for i in
                                     tqdm(self.X_train, ncols=100,
                                          desc="Resizing train images")])
            self.X_test = np.stack([imresize(i, resize) for i in
                                    tqdm(self.X_test, ncols=100,
                                         desc="Resizing test images")])

            if K.image_data_format() == "channels_first":
                # Bring channels back to the front
                self.X_train = np.transpose(self.X_train, (0, 3, 1, 2))
                self.X_test = np.transpose(self.X_test, (0, 3, 1, 2))

        self.X_train = self.X_train.astype(K.floatx()) / 255.
        self.X_test = self.X_test.astype(K.floatx()) / 255.

        self.y_train = to_categorical(self.y_train, 100)
        self.y_test = to_categorical(self.y_test, 100)

    @property
    def num_classes(self):
        """Number of output labels."""
        return 100


class Imdb(Dataset):

    """Imdb dataset with padding and truncation."""

    def __init__(self, num_words, max_seq_len, max_samples=None):
        super().__init__(lambda: imdb.load_data(num_words=num_words),
                         max_samples)
        self._num_words = num_words
        self._max_seq_len = max_seq_len
        self.X_train = pad_sequences(self.X_train, maxlen=max_seq_len)
        self.X_test = pad_sequences(self.X_test, maxlen=max_seq_len)

    @property
    def num_words(self):
        """Size of vocabulary."""
        return self._num_words

    @property
    def max_seq_len(self):
        """Maximum length of input sequences."""
        return self._max_seq_len


class Ptb(Dataset):

    """ PTB dataset with padding and truncation."""

    def __init__(self, batch_size, num_words=10000, num_steps=35,
                 max_samples=None):
        super().__init__(
            lambda: self.ptbloader(batch_size=batch_size,
                                   num_words=num_words,
                                   num_steps=num_steps),
            max_samples
        )
        self._num_words = num_words
        self._num_steps = num_steps

    def ptbloader(self, batch_size, num_words, num_steps):
        """Load PTB data and return as generators."""
        _, _, trn, val = ptb_raw_data("data/ptb")
        train_X, train_y = get_ptb(trn, batch_size, num_steps, num_words)
        val_X, val_y = get_ptb(val, batch_size, num_steps, num_words)
        return (train_X, train_y), (val_X, val_y)

    def get_batch_len(self, raw_data, batch_size, num_steps):
        """Returns the total number of batches in an epoch."""
        data = np.array(raw_data, dtype=np.int32)
        data_len = len(raw_data)
        nstep_len = data_len // num_steps
        return nstep_len // batch_size

    @property
    def num_words(self):
        """Size of vocabulary."""
        return self._num_words

    @property
    def num_steps(self):
        """Maximum length of input sequences."""
        return self._num_steps


class Babi(Dataset):

    """ bAbI-10k dataset."""

    def __init__(self, qnum, max_samples=None):
        super().__init__(
            lambda: self.loader(qnum),
            max_samples
        )

    def loader(self, qnum):
        """ Download / load the dataset."""
        try:
            path = get_file("babi-tasks-v1-2.tar.gz",
                            origin="https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz")
        except:
            print("Error downloading dataset")
            raise

        tar = tarfile.open(path)
        train = get_stories(tar.extractfile(QFILE[qnum]))
        vocab = sorted(reduce(
            lambda x, y: x | y,
            (set(story + q + [answer]) for story, q, answer in train)))

        self.vocab_size = len(vocab) + 1
        word_idx = dict((c, i+1) for i, c in enumerate(vocab))
        self.story_maxlen = max(map(len, (x for x, _, _ in train)))
        self.query_maxlen = max(map(len, (x for _, x, _ in train)))

        X, Xq, Y = vectorize_stories(train,
                                     word_idx,
                                     self.story_maxlen,
                                     self.query_maxlen)
        return ([X, Xq], Y), (None, None)
