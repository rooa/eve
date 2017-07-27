"""hypsearch.py: plot results of the hyperparameter sensitiviy experiment."""

import os
import pickle
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
    arg_parser.add_argument("--fig-size", type=float, nargs="+", required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    label_done = False
    with open(os.path.join(args.res_dir, "eve.pkl"), "rb") as f:
        d = pickle.load(f)
    df = d["losses_df"]
    for c in [2, 5, 10, 15, 20]:
        # Plot Eve results for this value of c
        for beta in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
            losses = list(df[np.isclose(df.beta, beta) &
                             np.isclose(df.c, c)].best_full_losses)[0]

            if beta == 0.999 and c == 10:
                ax.semilogy(range(1, 101), losses, color=OPT_COLORS["eve"],
                            linewidth=1, label="Eve (default)")
            else:
                if label_done:
                    ax.semilogy(range(1, 101), losses, color=OTHER_EVES_COLOR,
                                zorder=-1, linewidth=0.5)
                else:
                    label_done = True
                    ax.semilogy(range(1, 101), losses, color=OTHER_EVES_COLOR,
                                zorder=-1, linewidth=0.5, label="Eve (other)")

    # Plot Adam
    with open(os.path.join(args.res_dir, "adam.pkl"), "rb") as f:
        d = pickle.load(f)
    ax.semilogy(d["best_full_losses"], color=OPT_COLORS["adam"], label="Adam",
                linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    lgd = ax.legend(loc="upper right")
    for h in lgd.legendHandles:
        h.set_linewidth(1.5)

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    save_tight(fig, args.fig_size, args.save_path)


if __name__ == "__main__":
    main()
