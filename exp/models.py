"""models.py: definition of models for experiments."""

from abc import ABCMeta, abstractproperty

import numpy as np
from keras import layers
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                          LSTM, Embedding, BatchNormalization, Activation,
                          Input, AveragePooling2D, GRU, TimeDistributed,
                          RepeatVector, Merge)
from keras.regularizers import l1, l2
from keras.applications import ResNet50

from eve.exp.utils import im_bn_axis


class Model(metaclass=ABCMeta):

    """Wrapper around Keras model.

    Arguments:
        dataset: eve.exp.datasets.Dataset: the dataset on which the model
            will be run.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._model = None

    @property
    def model(self):
        """The Keras model wrapped by this object."""
        return self._model

    @abstractproperty
    def loss(self):
        """The loss function using which the model should be optimized."""
        raise NotImplementedError

    @property
    def loss_min(self):
        """The minimum (or its estimate) of the model loss function."""
        return 0


class WeightDecayCNN4Model(Model):

    """A weight-decay regularized multi-class classifiction CNN
    with 4 convolutional layers."""

    def __init__(self, dataset):
        super().__init__(dataset)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu",
                         input_shape=dataset.input_shape,
                         kernel_regularizer=l1(0.01)))
        model.add(Conv2D(32, (3, 3), activation="relu",
                         kernel_regularizer=l1(0.01)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu",
                         kernel_regularizer=l1(0.01)))
        model.add(Conv2D(64, (3, 3), activation="relu",
                         kernel_regularizer=l1(0.01)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.001)))
        model.add(Dense(dataset.num_classes, activation="softmax"))
        self._model = model

    @property
    def loss_min(self):
        # Estimate the minimum as the value of the regularization
        # at the initial weights
        weights = self._model.get_weights()
        loss_min_hat = 0.

        # Sum up the L1 penalties on the convolutional kernels
        for i in [0, 2, 4, 6]:
            loss_min_hat += 0.01 * np.sum(np.abs(weights[i]))

        # Add the L2 penalty on the dense layer
        loss_min_hat += 0.001 * np.sqrt(np.sum(np.square(weights[8])))

        return loss_min_hat

    @property
    def loss(self):
        return "categorical_crossentropy"


class DropoutCNN2Model(Model):

    """A dropout regularized multi-class classification CNN
    with 2 convolutional layers."""

    def __init__(self, dataset):
        super().__init__(dataset)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                         input_shape=dataset.input_shape))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(dataset.num_classes, activation="softmax"))
        self._model = model

    @property
    def loss(self):
        return "categorical_crossentropy"


class LSTM1Model(Model):

    """Single layer binary classification LSTM."""

    def __init__(self, dataset):
        super().__init__(dataset)
        model = Sequential()
        model.add(Embedding(dataset.num_words, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation="sigmoid"))
        self._model = model

    @property
    def loss(self):
        return "binary_crossentropy"


class GRU1Model(Model):

    """
    Single layer GRU for language modeling. The architecture is from
    TensorFlow's example on ptb language model.
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        model = Sequential()

        model.add(GRU(256, return_sequences=True,
                      input_shape=(dataset.num_steps, dataset.num_words)))
        model.add(Dropout(0.5))

        model.add(GRU(256, return_sequences=True))
        model.add(Dropout(0.5))

        model.add(TimeDistributed(Dense(dataset.num_words)))
        model.add(Activation("softmax"))

        self._model = model

    @property
    def loss(self):
        return "categorical_crossentropy"


class DropoutCNN4Model(Model):

    """A dropout regularized multi-class classification CNN
    with 4 convolutional layers."""

    def __init__(self, dataset):
        super().__init__(dataset)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                         input_shape=dataset.input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(dataset.num_classes, activation="softmax"))
        self._model = model

    @property
    def loss(self):
        return "categorical_crossentropy"


class ResnetModel(Model):

    """Deep residual network for multi-class classification."""

    @staticmethod
    def _conv_block(in_, filters):
        """A block with covolutional layer at shortcut.

        The first (and shortcut) conv layers perform 2x2 downsampling.

        Arguments:
            filters: 3-list: number of filters in the 3 conv layers.
        """
        x = Conv2D(filters[0], kernel_size=(1, 1), strides=(2, 2))(in_)
        x = BatchNormalization(axis=im_bn_axis())(x)
        x = Activation("relu")(x)

        x = Conv2D(filters[1], kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization(axis=im_bn_axis())(x)
        x = Activation("relu")(x)

        x = Conv2D(filters[2], kernel_size=(1, 1))(x)
        x = BatchNormalization(axis=im_bn_axis())(x)

        shortcut = Conv2D(filters[2], kernel_size=(1, 1),
                          strides=(2, 2))(in_)
        shortcut = BatchNormalization(axis=im_bn_axis())(shortcut)

        x = layers.add([x, shortcut])
        x = Activation("relu")(x)
        return x

    @staticmethod
    def _identity_block(in_, filters):
        """A block without convolutional layer at shortcut.

        Arguments:
            filters: 3-list: number of filters in the 3 conv layers.
        """
        x = Conv2D(filters[0], kernel_size=(1, 1))(in_)
        x = BatchNormalization(axis=im_bn_axis())(x)
        x = Activation("relu")(x)

        x = Conv2D(filters[1], kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization(axis=im_bn_axis())(x)
        x = Activation("relu")(x)

        x = Conv2D(filters[2], kernel_size=(1, 1))(x)
        x = BatchNormalization(axis=im_bn_axis())(x)

        x = layers.add([x, in_])
        x = Activation("relu")(x)
        return x

    def __init__(self, dataset):
        super().__init__(dataset)
        in_ = Input(shape=dataset.input_shape)
        x = Conv2D(32, kernel_size=(3, 3))(in_)
        x = BatchNormalization(axis=im_bn_axis())(x)
        x = Activation("relu")(x)

        x = ResnetModel._conv_block(x, [32, 32, 128])
        x = ResnetModel._identity_block(x, [32, 32, 128])

        x = ResnetModel._conv_block(x, [64, 64, 256])
        x = ResnetModel._identity_block(x, [64, 64, 256])
        x = ResnetModel._identity_block(x, [64, 64, 256])

        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x = Dense(dataset.num_classes, activation="softmax")(x)

        self._model = KerasModel(inputs=in_, outputs=x)

    @property
    def loss(self):
        return "categorical_crossentropy"


class Resnet50Model(Model):

    """The original Resnet-50 model."""

    def __init__(self, dataset):
        super().__init__(dataset)
        in_ = Input(shape=dataset.input_shape)
        base_model = ResNet50(include_top=False, weights=None, input_tensor=in_,
                              input_shape=dataset.input_shape, pooling=None)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(dataset.num_classes, activation="softmax")(x)

        self._model = KerasModel(inputs=in_, outputs=x)

    @property
    def loss(self):
        return "categorical_crossentropy"


class BABIModel(Model):

    """ The RNN model for bAbI QA task. """

    def __init__(self, dataset):
        super().__init__(dataset)
        vocab_size = dataset.vocab_size
        story_maxlen = dataset.story_maxlen
        query_maxlen = dataset.query_maxlen

        sentrnn = Sequential()
        sentrnn.add(Embedding(vocab_size, 256,
                              input_length=story_maxlen))
        sentrnn.add(Dropout(0.3))

        qrnn = Sequential()
        qrnn.add(Embedding(vocab_size, 256,
                           input_length=query_maxlen))
        qrnn.add(Dropout(0.3))
        qrnn.add(GRU(256, return_sequences=False))
        qrnn.add(RepeatVector(story_maxlen))

        model = Sequential()
        model.add(Merge([sentrnn, qrnn], mode="sum"))
        model.add(GRU(256, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(vocab_size, activation="softmax"))

        self._model = model

    @property
    def loss(self):
        return "categorical_crossentropy"
