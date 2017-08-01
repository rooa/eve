"""main.py: runner for hyperparameter sensitivity experiment.

This file runs Eve over a grid of (beta, c) values, and collects
the results.
"""

import itertools
import json
from argparse import ArgumentParser

import numpy as np
from eve.exp.runners.utils import EXP_SEED
np.random.seed(EXP_SEED)

import pandas as pd

from eve.optim.eve import Eve
from eve.exp.datasets import Dataset
from eve.exp.models import Model
from eve.exp.callbacks import EpochFullLossHistory
from eve.exp.utils import (build_subclass_object, get_subclass_names,
                           get_subclass_from_name, save_pkl)


def main():
    """Run the hyperparameter sensitivity experiment."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model", type=str, required=True,
                            choices=get_subclass_names(Model))
    arg_parser.add_argument("--dataset", type=str, required=True,
                            choices=get_subclass_names(Dataset))
    arg_parser.add_argument("--dataset-kwargs", type=json.loads, default="{}")
    arg_parser.add_argument("--lrs", type=float, required=True, nargs="+")
    arg_parser.add_argument("--batch-size", type=int, required=True)
    arg_parser.add_argument("--epochs", type=int, required=True)
    arg_parser.add_argument("--betas", type=float, required=True, nargs="+")
    arg_parser.add_argument("--cs", type=float, required=True, nargs="+")
    arg_parser.add_argument("--save-path", type=str, required=True)
    args = arg_parser.parse_args()

    # Load data
    dataset = build_subclass_object(Dataset, args.dataset, args.dataset_kwargs)
    X, y = dataset.train_data

    # Loop over beta and c values
    losses = []
    for beta, c in itertools.product(args.betas, args.cs):
        # Loop over lr values to find the best
        best_final_loss = None
        for lr in args.lrs:
            print("beta {}, c {}, lr {}".format(beta, c, lr))

            # Set up model according to beta, c, and lr
            model = get_subclass_from_name(Model, args.model)(dataset)
            opt = Eve(lr=lr, beta_3=beta, c=c, loss_min=model.loss_min)
            model.model.compile(loss=model.loss, optimizer=opt)

            # Train
            full_loss_history = EpochFullLossHistory(X, y, args.batch_size)
            model.model.fit(X, y, batch_size=args.batch_size,
                            epochs=args.epochs, callbacks=[full_loss_history])

            full_losses = full_loss_history.losses
            if best_final_loss is None or full_losses[-1] < best_final_loss:
                best_final_loss = full_losses[-1]
                best_full_losses = full_losses
                best_lr = lr

        # Store summary for this beta, c
        losses.append({"beta": beta, "c": c, "best_lr": best_lr,
                       "best_full_losses": best_full_losses})
        print()

    losses_df = pd.DataFrame(losses)
    save_data = {"cmd_args": args, "losses_df": losses_df}
    save_pkl(save_data, args.save_path)


if __name__ == "__main__":
    main()
