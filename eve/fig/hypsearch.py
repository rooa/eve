"""hypsearch.py: plot results of the hyperparameter sensitiviy experiment."""

import os
import pickle
from glob import glob
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eve.fig.utils import OPT_COLORS, sns_set_defaults, save_tight

OTHER_EVES_COLOR = [255./255., 194./255, 255./255]


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--res-dir", type=str, required=True)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--ylim", type=float, nargs=2)
    arg_parser.add_argument("--no-logscale", action="store_true")
    arg_parser.add_argument("--add-legend", action="store_true")
    arg_parser.add_argument("--xticks", type=float, nargs="+")
    arg_parser.add_argument("--yticks", type=float, nargs="+")
    arg_parser.add_argument("--context", type=str, default="paper")
    arg_parser.add_argument("--font", type=str, default=None)
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks", font=args.font)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_fun = ax.plot if args.no_logscale else ax.semilogy

    label_done = False
    for eve_file in glob(os.path.join(args.res_dir, "*.pkl")):
        with open(eve_file, "rb") as f:
            d = pickle.load(f)
        df = d["losses_df"]

        # Plot Eve results for this file
        for _, row in df.iterrows():
            losses = row.best_full_losses
            if np.isclose(row.beta, 0.999) and np.isclose(row.c, 10):
                plot_fun(range(1, len(losses)+1), losses,
                         color=OPT_COLORS["eve"], linewidth=1,
                         label="Eve (default)")
            else:
                if label_done:
                    plot_fun(range(1, len(losses)+1), losses,
                             color=OTHER_EVES_COLOR, zorder=-1, linewidth=0.5)
                else:
                    label_done = True
                    plot_fun(range(1, len(losses)+1), losses,
                             color=OTHER_EVES_COLOR, zorder=-1, linewidth=0.5,
                             label="Eve (other)")

    # Plot Adam
    exp = args.res_dir.split("/")[-1]
    with open(os.path.join("data", "results", "trainloss", exp, "adam.pkl"),
              "rb") as f:
        d = pickle.load(f)
    plot_fun(d["best_full_losses"], color=OPT_COLORS["adam"], label="Adam",
             linewidth=1)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_xlim(0, len(d["best_full_losses"]))

    if args.xticks is not None:
        ax.set_xticks(args.xticks)

    if args.yticks is not None:
        ax.set_yticks(args.yticks)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if args.add_legend:
        lgd = ax.legend(loc="upper right")
        for h in lgd.legendHandles:
            h.set_linewidth(1.5)

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    save_tight(fig, args.fig_size, args.save_path)


if __name__ == "__main__":
    main()
