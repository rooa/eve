"""lreff.py: plot effective learning rates."""

import pickle
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from eve.fig.utils import sns_set_defaults, save_tight


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs="+", required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    args = arg_parser.parse_args()

    sns_set_defaults(args.context, style="ticks")
    sns.set_palette(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tasks = ["resnet", "ptb/noembed", "imdb", "babi/q14"]
    task_names = ["CIFAR 100", "PTB", "IMDB", "BABI"]

    for i in range(4):
        with open(os.path.join("data/results/trainloss", tasks[i],
                               "eve.pkl"), "rb") as f:
            data = pickle.load(f)
        ax.semilogy(data["best_lr_effs"], label=task_names[i])

    ax.set_yticks([1e-4, 1e-3, 1e-2])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Effective learning rate ($\alpha_1 / d_t$)")

    lgd = ax.legend(loc="upper right")
    for h in lgd.legendHandles:
        h.set_linewidth(1.5)

    sns.despine(fig, ax, top=True, right=True, bottom=False, left=False)
    ax.tick_params(axis="y", which="minor", left="off")

    save_tight(fig, args.fig_size, args.save_path)


if __name__ == "__main__":
    main()
