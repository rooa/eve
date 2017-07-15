"""callbacks.py: Keras callback functions used for experiments."""


from keras.callbacks import Callback


class BatchLossHistory(Callback):

    """Callback to record loss after each batch."""

    def on_train_begin(self, logs=None):
        """Initialize batch losses list."""
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        """Store the previous batch's loss."""
        if logs is not None:
            self.batch_losses.append(logs["loss"].item())


class EpochFullLossHistory(Callback):

    """Callback to record loss over provided data at the
    end of each epoch."""

    def __init__(self, X, y, batch_size):
        """Initialize the callback object.

        Arguments:
            X, y: numpy arrays: data to be used for computing loss.
            batch_size: int: batch size used when computing loss.
        """
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        """Initialize losses list."""
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        """Compute full loss and store it."""

        eval_res = self.model.evaluate(self.X, self.y, self.batch_size,
                                       verbose=0)
        if type(eval_res) == list:
            self.losses.append(eval_res[0])
        else:
            self.losses.append(eval_res)
