"""utils.py: utility functions for plotting figures."""

import os


OPT_COLORS = {
    "eve": "#e7298a",
    "adam": "#e6ab02",
    "adamax": "#7570b3",
    "rmsprop": "#66a61e",
    "adagrad": "#d95f02",
    "adadelta": "#a6761d",
    "sgd": "#1b9e77",
}
"""Colors used to represent the different optimizers."""


OPT_LABELS = {
    "eve": "Eve",
    "adam": "Adam",
    "adamax": "Adamax",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "sgd": "SGD",
}
"""Names to be used for describing optimizers in legends."""


class Fig:

    """Wrapper around matplotlib figure."""

    RC_DEFAULT = {
        # Configure pgf
        "backend": "pgf",
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False,  # Fonts are manually specified
        "pgf.preamble": [
            r"\usepackage[no-math]{fontspec}",
            r"\usepackage[dvipsnames]{xcolor}",
            r"\setmainfont{futura_book.ttf}[Path=\string~/.fonts/]",
        ],
        # Configure text
        "text.color": "k",
        "text.usetex": True,
        # Configure lines
        "lines.linewidth": 1,
        "lines.color": "k",
        "lines.markersize": 15,
        # Configure ticks
        "xtick.direction": "in",
        "xtick.color": "k",
        "xtick.major.size": 2,
        "xtick.major.width": 1,
        "xtick.major.pad": 5,
        "ytick.direction": "in",
        "ytick.color": "k",
        "ytick.major.size": 2,
        "ytick.major.width": 1,
        "ytick.major.pad": 2,
        # Configure axes
        "axes.linewidth": 1,
        "axes.edgecolor": "k",
        "axes.labelcolor": "k",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.axisbelow": True,
        # Configure legend
        "legend.frameon": False,
    }

    def __init__(self, context="paper", style="ticks", **rc_extra):
        rc = Fig.RC_DEFAULT.copy()
        rc.update(rc_extra)

        import matplotlib as mpl

        mpl.rcParams.update(rc)

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(context=context, style=style, rc=rc)
        self.fig = plt.figure()
        self.mpl = mpl
        self.plt = plt
        self.sns = sns

    def save(self, size, save_dir, name, bbox_extra_artists=None):
        """Save figure to the given path as a pdf image."""
        self.fig.set_size_inches(*size)
        self.fig.tight_layout(
            pad=0, w_pad=0, h_pad=0, rect=(0.05, 0.05, 0.95, 0.95)
        )
        self.fig.savefig(
            os.path.join(save_dir, f"{name}.pdf"),
            bbox_extra_artists=bbox_extra_artists,
        )
