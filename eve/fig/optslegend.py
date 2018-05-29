"""optslegend.py: create a legend with all the optimizers."""

from argparse import ArgumentParser

from eve.fig.utils import OPT_COLORS, OPT_LABELS, Fig


def main():
    """Create and save the legend."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--save-dir", type=str, required=True)
    arg_parser.add_argument("--fig-name", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    fig = Fig(args.context)
    ax = fig.fig.add_subplot(111)

    # Plot a dummy figure to get a legend
    opts = ["eve", "adam", "adamax", "rmsprop", "adagrad", "adadelta", "sgd"]
    for opt in opts:
        ax.scatter([1], [1], color=OPT_COLORS[opt], label=OPT_LABELS[opt])
    ax.legend()

    # Copy the legend to a new figure
    lhandles, llabels = ax.get_legend_handles_labels()
    fig = Fig(args.context)
    ax = fig.fig.add_subplot(111)
    ax.legend(
        lhandles,
        llabels,
        ncol=len(opts),
        scatterpoints=1,
        loc="center",
        markerscale=1,
        mode="expand",
        borderpad=0,
        borderaxespad=0,
    )
    fig.sns.despine(
        fig.fig, ax, top=True, bottom=True, left=True, right=True, trim=True
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.save(args.fig_size, args.save_dir, args.fig_name)


if __name__ == "__main__":
    main()
