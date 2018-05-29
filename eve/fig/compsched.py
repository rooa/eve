"""cmpsched.py: plot results of comparing Eve aginst learning rate schedules."""

import os
import pickle
from glob import glob
from argparse import ArgumentParser

from eve.fig.utils import OPT_COLORS, Fig


def main():
    """Create and save the figure."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--res-dir", type=str, required=True)
    arg_parser.add_argument("--save-dir", type=str, required=True)
    arg_parser.add_argument("--fig-size", type=float, nargs=2, required=True)
    arg_parser.add_argument("--context", type=str, default="paper")
    arg_parser.add_argument("--no-logscale", action="store_true")
    arg_parser.add_argument("--ylim", type=float, nargs=2)
    arg_parser.add_argument("--xticks", type=float, nargs="+")
    arg_parser.add_argument("--yticks", type=float, nargs="+")
    args = arg_parser.parse_args()

    fig = Fig(args.context)
    ax = fig.fig.add_subplot(111)

    plot_fun = ax.plot if args.no_logscale else ax.semilogy
    experiment = args.res_dir.split("/")[-1]

    with open(
        os.path.join("data", "results", "trainloss", experiment, "eve.pkl"),
        "rb",
    ) as f:
        d = pickle.load(f)
    epochs = len(d["best_full_losses"])
    plot_fun(
        range(1, epochs + 1),
        d["best_full_losses"],
        color=OPT_COLORS["eve"],
        label="Eve",
        zorder=2,
    )

    best_exp_loss = None
    for exp_dec_file in glob(os.path.join(args.res_dir, "exps", "*.pkl")):
        with open(exp_dec_file, "rb") as f:
            d = pickle.load(f)
        if best_exp_loss is None or d["best_full_losses"][-1] < best_exp_loss:
            best_exp_loss = d["best_full_losses"][-1]
            best_exp_dec_file = exp_dec_file
            best_exp_dec = d["best_decay"]

    with open(best_exp_dec_file, "rb") as f:
        d = pickle.load(f)
    plot_fun(
        range(1, epochs + 1),
        d["best_full_losses"],
        color="#ff7f00",
        label=r"$e^{-t}$ decay",
        zorder=2,
    )

    decs = ["inv", "sqrt"]
    dec_titles = ["$1/t$ decay", r"$1/\sqrt{t}$ decay"]
    dec_colors = ["#e41a1c", "#999999"]
    for dec, dec_title, dec_color in zip(decs, dec_titles, dec_colors):
        with open(os.path.join(args.res_dir, "{}.pkl".format(dec)), "rb") as f:
            d = pickle.load(f)
        plot_fun(
            range(1, epochs + 1),
            d["best_full_losses"],
            color=dec_color,
            label=dec_title,
            zorder=2,
        )

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_xlim(0, epochs)

    if args.xticks is not None:
        ax.set_xticks(args.xticks)

    if args.yticks is not None:
        ax.set_yticks(args.yticks)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.sns.despine(fig.fig, ax, top=True, right=True, bottom=False, left=True)
    ax.tick_params(axis="y", which="minor", left="off")
    ax.tick_params(axis="y", which="major", length=0)
    ax.grid(b=True, axis="y", which="major", zorder=0)
    ax.legend(
        loc="upper right",
        facecolor="white",
        framealpha=1,
        frameon=True,
        edgecolor="white",
        bbox_to_anchor=(1.1, 1.025),
    ).set_zorder(
        1
    )

    fig.save(args.fig_size[:2], args.save_dir, "bests")

    fig = Fig(args.context)
    ax = fig.fig.add_subplot(111)
    plot_fun = ax.plot if args.no_logscale else ax.semilogy

    sm_label_done, lg_label_done = False, False
    for exp_dec_file in glob(os.path.join(args.res_dir, "exps", "*.pkl")):
        with open(exp_dec_file, "rb") as f:
            d = pickle.load(f)

        if d["best_decay"] < best_exp_dec:
            if sm_label_done:
                plot_fun(
                    range(1, epochs + 1),
                    d["best_full_losses"],
                    color="#1f78b4",
                    zorder=2,
                )
            else:
                sm_label_done = True
                plot_fun(
                    range(1, epochs + 1),
                    d["best_full_losses"],
                    color="#1f78b4",
                    label="Smaller decay",
                    zorder=2,
                )

        elif d["best_decay"] > best_exp_dec:
            if lg_label_done:
                plot_fun(
                    range(1, epochs + 1),
                    d["best_full_losses"],
                    color="#fb9a99",
                    zorder=2,
                )
            else:
                lg_label_done = True
                plot_fun(
                    range(1, epochs + 1),
                    d["best_full_losses"],
                    color="#fb9a99",
                    label="Larger decay",
                    zorder=2,
                )

        else:
            plot_fun(
                range(1, epochs + 1),
                d["best_full_losses"],
                color="#ff7f00",
                label="Optimal decay",
                zorder=2,
            )

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_xlim(0, epochs)

    if args.xticks is not None:
        ax.set_xticks(args.xticks)

    if args.yticks is not None:
        ax.set_yticks(args.yticks)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.sns.despine(fig.fig, ax, top=True, right=True, bottom=False, left=True)
    ax.tick_params(axis="y", which="minor", left="off")
    ax.tick_params(axis="y", which="major", length=0)
    ax.grid(b=True, axis="y", which="major", zorder=0)
    ax.legend(
        loc="upper right",
        facecolor="white",
        framealpha=1,
        frameon=True,
        edgecolor="white",
        bbox_to_anchor=(1.1, 1),
    ).set_zorder(
        1
    )

    fig.save(args.fig_size[:2], args.save_dir, "exps")


if __name__ == "__main__":
    main()
