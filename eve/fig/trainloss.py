"""trainloss.py: plot results of training loss experiments."""

import pickle
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from eve.fig.utils import OPT_COLORS, OPT_LABELS, sns_set_defaults, save_tight


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--res-dir", type=str, required=True)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    arg_parser.add_argument("--add-legend", action="store_true")
    arg_parser.add_argument("--opts", type=str, nargs="+",
                            default=["adadelta", "adagrad", "rmsprop", "adamax",
                                     "adam", "sgd", "eve"])
    arg_parser.add_argument("--no-logscale", action="store_true")
    arg_parser.add_argument("--ylim", type=float, nargs=2)
    arg_parser.add_argument("--inset", action="store_true")
    arg_parser.add_argument("--xticks", type=float, nargs="+")
    arg_parser.add_argument("--yticks", type=float, nargs="+")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_fun = ax.plot if args.no_logscale else ax.semilogy

    for opt in args.opts:
        with open(os.path.join(args.res_dir, "{}.pkl".format(opt)), "rb") as f:
            opt_data = pickle.load(f)
        losses = opt_data["best_full_losses"]
        plot_fun(range(1, len(losses)+1), losses, color=OPT_COLORS[opt],
                 label=OPT_LABELS[opt], zorder=-1)

    # Label axes
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if args.xticks is not None:
        ax.set_xticks(args.xticks)

    if args.yticks is not None:
        ax.set_yticks(args.yticks)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_xlim(0, len(losses))

    # Create legend
    if args.add_legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), handlelength=1)
        for h in lgd.legendHandles:
            h.set_linewidth(1.5)

        bbox_extra_artists = (lgd,)
        fig_rect = (0.05, 0.05, 0.75, 0.95)
    else:
        bbox_extra_artists = None
        fig_rect = None

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    # Create inset for ptb
    if args.inset:
        iax = fig.add_axes([.65, .5, .25, .3])

        for opt in args.opts:
            with open(os.path.join(args.res_dir, "{}.pkl".format(opt)), "rb") \
                    as f:
                opt_data = pickle.load(f)
            losses = opt_data["best_full_losses"]
            iax.plot(losses[-10:], color=OPT_COLORS[opt], label=OPT_LABELS[opt])

        iax.set_ylim(1, 1.04)

        iax.set_xticks([])
        iax.set_yticks([])

        for axis in ["top", "bottom", "left", "right"]:
            iax.spines[axis].set_linewidth(0.5)

        ax.add_patch(mpatches.Rectangle((90, 1), 10, 0.05, fill=False,
                                        zorder=10))
        ax.annotate(s="", xy=(80, 1.30), xytext=(95, 1.05),
                    arrowprops={
                        "arrowstyle": "-|>,head_length=0.3,head_width=0.1",
                        "connectionstyle": "arc3,rad=0",
                        "linewidth": 0.5,
                        "facecolor": "k"
                    })

    save_tight(fig, args.fig_size, args.save_path,
               bbox_extra_artists=bbox_extra_artists, fig_rect=fig_rect)


if __name__ == "__main__":
    main()
