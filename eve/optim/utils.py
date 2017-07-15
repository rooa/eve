"""utils.py: utility functions."""

import numpy as np
import keras.backend as K


def fmin_pos(dtype):
    """Return the smallest positive number representable
    in the given data type.

    Arguments:
        dtype: a numpy datatype like "float32", "float64" etc.
    """
    return np.nextafter(np.cast[dtype](0), np.cast[dtype](1))


def fmin_pos_floatx():
    """Return the smallest positive number representable
    using the Keras floatX data type.
    """
    return fmin_pos(K.floatx())
