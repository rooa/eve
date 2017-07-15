"""cmpexps.py: plot results comparing Eve against various exponential decays."""

import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eve.fig.utils import OPT_COLORS, sns_set_defaults, save_tight


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs="+", required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    with open("data/results/lrsched/resnet/eve.pkl", "rb") as f:
        data = pickle.load(f)
    ax.semilogy(range(1, 101), data["best_full_losses"],
                color=OPT_COLORS["eve"], label="Eve")

    adam_colors = [[161, 218, 180], [65, 182, 196], [44, 127, 184],
                   [37, 52, 148], [103, 44, 0]]
    adam_colors = np.array(adam_colors) / 255.

    decays = ["0.5", "1", "2", "4", "8"]
    for i in range(5):
        with open(
            "data/results/lrsched/resnet/adam_exp_lr1e-3_decay{}e-5.pkl".format(
                decays[i]),
            "rb"
        ) as f:
            data = pickle.load(f)
        ax.semilogy(range(1, 101), data["best_full_losses"],
                    color=adam_colors[i],
                    label=r"Adam (exp.: ${}$)".format(decays[i]))

    ax.semilogy([], [], color=[255./255., 197./255., 26./255.],
                label="Adam (1/t decay)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for h in lgd.legendHandles:
        h.set_linewidth(1.5)
    bbox_extra_artists = (lgd,)

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    save_tight(fig, args.fig_size, args.save_path,
               bbox_extra_artists=bbox_extra_artists)


if __name__ == "__main__":
    main()
