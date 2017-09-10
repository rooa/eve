"""optslegend.py: create a legend with all the optimizers."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from eve.fig.utils import OPT_COLORS, OPT_LABELS, sns_set_defaults, save_tight


def main():
    """Create and save the legend."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot a dummy figure to get a legend
    opts = ["eve", "adam", "adamax", "rmsprop", "adagrad", "adadelta", "sgd"]
    for opt in opts:
        ax.scatter([1], [1], color=OPT_COLORS[opt], label=OPT_LABELS[opt])
    ax.legend()

    # Copy the legend to a new figure
    lhandles, llabels = ax.get_legend_handles_labels()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.legend(lhandles, llabels, ncol=7, scatterpoints=1, loc="center",
              markerscale=1, mode="expand", borderpad=0, borderaxespad=0)
    sns.despine(fig, ax, top=True, bottom=True, left=True, right=True,
                trim=True)
    ax.set_xticks([])
    ax.set_yticks([])

    save_tight(fig, args.fig_size, args.save_path)


if __name__ == "__main__":
    main()
