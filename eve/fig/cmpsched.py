"""cmpsched.py: plot results of comparing Eve aginst learning rate schedules."""

import os
import pickle
from glob import glob
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eve.fig.utils import OPT_COLORS, sns_set_defaults, save_tight


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--res-dir", type=str, required=True)
    arg_parser.add_argument("--adam-best-lr", type=str, required=True)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs="+", required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(121)

    with open(os.path.join(args.res_dir, "eve.pkl"), "rb") as f:
        data = pickle.load(f)
    ax.semilogy(range(1, 101), data["best_full_losses"],
                color=OPT_COLORS["eve"], label="Eve")

    best_full_losses = {}
    best_lrs = {}
    for dec in ["exp", "inv"]:
        best_loss = np.inf
        for fname in glob(os.path.join(args.res_dir, "adam_{}*".format(dec))):
            with open(fname, "rb") as f:
                d = pickle.load(f)
            if d["best_full_losses"][-1] < best_loss:
                best_loss = d["best_full_losses"][-1]
                best_full_losses[dec] = d["best_full_losses"]
                best_lrs[dec] = d["best_lr"]

    print("best_lrs: {}".format(best_lrs))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.semilogy(range(1, 101), best_full_losses["exp"],
                label="Adam (exp. decay)",
                color=[103./255., 44./255., 0./255.])
    ax.semilogy(range(1, 101), best_full_losses["inv"],
                label="Adam ($1/t$ decay)",
                color=[255./255., 197./255., 26./255.])

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    ax = fig.add_subplot(122)

    with open(os.path.join(args.res_dir, "eve.pkl"), "rb") as f:
        data = pickle.load(f)
    ax.semilogy(range(1, 101), data["best_full_losses"],
                color=OPT_COLORS["eve"], label="Eve")

    adam_colors = [[161, 218, 180], [65, 182, 196], [44, 127, 184],
                   [37, 52, 148], [103, 44, 0]]
    adam_colors = np.array(adam_colors) / 255.

    decays = ["0.5", "1", "2", "4", "8"]
    for i in range(5):
        with open(
            os.path.join(args.res_dir, "adam_exp_lr{}_decay{}e-5.pkl".format(
                args.adam_best_lr, decays[i])),
            "rb"
        ) as f:
            data = pickle.load(f)
        ax.semilogy(range(1, 101), data["best_full_losses"],
                    color=adam_colors[i],
                    label=r"Adam (exp.: ${}$)".format(decays[i]))

    ax.semilogy([], [], color=[255./255., 197./255., 26./255.],
                label="Adam ($1/t$ decay)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for h in lgd.legendHandles:
        h.set_linewidth(1.5)
    bbox_extra_artists = (lgd,)

    # sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", which="minor", left="off")

    save_tight(fig, args.fig_size, args.save_path,
               bbox_extra_artists=bbox_extra_artists)


if __name__ == "__main__":
    main()
