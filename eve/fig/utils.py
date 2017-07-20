"""utils.py: utility functions for plotting figures."""

import seaborn as sns


OPT_COLORS = {
    "eve": "#e7298a",
    "adam": "#e6ab02",
    "adamax": "#7570b3",
    "rmsprop": "#66a61e",
    "adagrad": "#d95f02",
    "adadelta": "#a6761d",
    "sgd": "#1b9e77"
}
"""Colors used to represent the different optimizers."""


OPT_LABELS = {
    "eve": "Eve",
    "adam": "Adam",
    "adamax": "Adamax",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "sgd": "SGD"
}
"""Names to be used for describing optimizers in legends."""


def sns_set_defaults(context, style):
    """Set default values for Seaborn."""
    sns.set(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": True,
            "text.usetex": True,
            "font.serif": [],
            "lines.linewidth": 0.85,
            "xtick.direction": "inout",
            "xtick.major.size": 4,
            "xtick.major.pad": 2,
            "ytick.direction": "inout",
            "ytick.major.size": 4,
            "ytick.major.pad": 2,
            "xtick.color": "k",
            "ytick.color": "k",
            "axes.edgecolor": "k",
            "axes.labelcolor": "k",
        }
    )


def save_tight(fig, size, save_path, bbox_extra_artists=None):
    """Set a figure to tight layout, and save.

    Arguments:
        fig: matplotlib figure: the figure to save.
        size: 2-tuple: size of the figure in inches.
        save_path: str: full path of the save location.
        bbox_extra_artists: tuple: extra artists to be passed to save function.
    """
    fig.set_size_inches(*size)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    if bbox_extra_artists is None:
        fig.savefig(save_path)
    else:
        fig.savefig(save_path, bbox_inches="tight",
                    bbox_extra_artists=bbox_extra_artists)
