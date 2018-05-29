"""hypsearch.py: plot results of the hyperparameter sensitiviy experiment."""

import os
import pickle
from glob import glob
from argparse import ArgumentParser

import numpy as np

from eve.fig.utils import OPT_COLORS, Fig

OTHER_EVES_COLOR = [255. / 255., 194. / 255, 255. / 255]


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--res-dir", type=str, required=True)
    arg_parser.add_argument("--save-dir", type=str, required=True)
    arg_parser.add_argument("--fig-name", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--ylim", type=float, nargs=2)
    arg_parser.add_argument("--no-logscale", action="store_true")
    arg_parser.add_argument("--add-legend", action="store_true")
    arg_parser.add_argument("--xticks", type=float, nargs="+")
    arg_parser.add_argument("--yticks", type=float, nargs="+")
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    fig = Fig(args.context)
    ax = fig.fig.add_subplot(111)

    plot_fun = ax.plot if args.no_logscale else ax.semilogy

    label_done = False
    for eve_file in glob(os.path.join(args.res_dir, "*.pkl")):
        with open(eve_file, "rb") as f:
            d = pickle.load(f)
        df = d["losses_df"]

        # Plot Eve results for this file
        for _, row in df.iterrows():
            losses = row["best_full_losses"]
            if np.isclose(row.beta, 0.999) and np.isclose(row.c, 10):
                plot_fun(
                    range(1, len(losses) + 1),
                    losses,
                    color=OPT_COLORS["eve"],
                    label="Eve (default)",
                    zorder=3,
                )
            else:
                if label_done:
                    plot_fun(
                        range(1, len(losses) + 1),
                        losses,
                        color=OTHER_EVES_COLOR,
                        zorder=2,
                        linewidth=0.5,
                    )
                else:
                    label_done = True
                    plot_fun(
                        range(1, len(losses) + 1),
                        losses,
                        color=OTHER_EVES_COLOR,
                        zorder=2,
                        linewidth=0.5,
                        label="Eve (other)",
                    )

    # Plot Adam
    exp = args.res_dir.split("/")[-1]
    with open(
        os.path.join("data", "results", "trainloss", exp, "adam.pkl"), "rb"
    ) as f:
        d = pickle.load(f)
    plot_fun(
        d["best_full_losses"], color=OPT_COLORS["adam"], label="Adam", zorder=3
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if args.xticks is not None:
        ax.set_xticks(args.xticks)

    if args.yticks is not None:
        ax.set_yticks(args.yticks)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_xlim(0, len(losses))

    if args.add_legend:
        lgd = ax.legend(
            loc="upper right",
            facecolor="white",
            framealpha=1,
            frameon=True,
            edgecolor="white",
            bbox_to_anchor=(1.1, 1.075),
        )
        lgd.set_zorder(1)
        for h in lgd.legendHandles:
            h.set_linewidth(2)

    fig.sns.despine(fig.fig, ax, top=True, right=True, bottom=False, left=True)
    ax.tick_params(axis="y", which="minor", left="off")
    ax.tick_params(axis="y", which="major", length=0)
    ax.grid(b=True, axis="y", which="major", zorder=0)

    fig.save(args.fig_size, args.save_dir, args.fig_name)


if __name__ == "__main__":
    main()
