"""monitor.py: callback for Eve that tracks internals of the optimizer."""

from keras.callbacks import Callback
from keras import backend as K


class EveMonitor(Callback):

    """Callback for monitoring Eve behavior."""

    def on_train_begin(self, logs=None):
        """Initialize arrays."""
        self.d_nums = []
        self.d_dens = []
        self.ds = []
        self.lr_effs = []

    def on_batch_end(self, batch, logs=None):
        """Get latest values of optimizer internals."""
        self.d_nums.append(K.get_value(self.model.optimizer.d_num).item())
        self.d_dens.append(K.get_value(self.model.optimizer.d_den).item())
        self.ds.append(K.get_value(self.model.optimizer.d).item())
        self.lr_effs.append(K.get_value(self.model.optimizer.lr_eff).item())

    def get_data(self):
        """Get arrays in the monitor packed in a dictionary."""
        return {
            "d_nums": self.d_nums,
            "d_dens": self.d_dens,
            "ds": self.ds,
            "lr_effs": self.lr_effs
        }
