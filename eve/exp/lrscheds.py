"""lrscheds.py: learning rate schedules."""

from abc import ABCMeta, abstractmethod

import numpy as np
from keras.callbacks import Callback
from keras import backend as K


class LRSchedule(Callback, metaclass=ABCMeta):

    """Generic learning rate schedule.

    Arguments:
        lr_init: float: initial learning rate.
        decay: float: decay strength.
    """

    def __init__(self, lr_init, decay):
        super().__init__()
        self.lr_init = lr_init
        self.decay = decay
        self.iters = 0

    def on_batch_end(self, batch, logs=None):
        """Update learning rate."""
        self.iters += 1
        K.set_value(self.model.optimizer.lr, self.get_lr())

    @abstractmethod
    def get_lr(self):
        """Get learning rate based on the current number of iterations."""
        raise NotImplementedError


class InverseTimeDecayLRSchedule(LRSchedule):

    """Inverse time decay learning rate schedule.

    The learning rate at iteration t is computed as
    lr_init / (1 + decay*t).
    """

    def get_lr(self):
        return self.lr_init / (1. + self.decay*self.iters)


class ExponentialDecayLRSchedule(LRSchedule):

    """Exponential decay learning rate schedule.

    The learning rate at iteration t is computed as
    lr_init * exp(-decay*t).
    """

    def get_lr(self):
        return self.lr_init * np.exp(-self.decay*self.iters)


class SqrtTimeDecaySchedule(LRSchedule):

    """Inverse square root time decay learning rate schedule.

    The learning rate at iteration t is computed as
    lr_init / sqrt(1 + decay*t).
    """

    def get_lr(self):
        return self.lr_init / np.sqrt(1. + self.decay*self.iters)
