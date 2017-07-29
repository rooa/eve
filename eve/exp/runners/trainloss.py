"""trainloss.py: runner for minimizing training loss of a model.

This file runs an optimizer over a model, with the objective of reducing
the training loss as much as possible. Multiple learning rates can be
specified, and the one which leads to the lowest loss is selected.
"""

import json

from argparse import ArgumentParser

import numpy as np
np.random.seed(615879206)

from keras.optimizers import Optimizer

from eve.optim.eve import Eve
from eve.optim.monitor import EveMonitor
from eve.exp.datasets import Dataset
from eve.exp.models import Model
from eve.exp.callbacks import BatchLossHistory, EpochFullLossHistory
from eve.exp.utils import (build_subclass_object, get_subclass_names,
                           get_subclass_from_name, save_pkl)


def main():
    """Run experiment for training loss optimization."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--opt", type=str, required=True,
                            choices=get_subclass_names(Optimizer))
    arg_parser.add_argument("--opt-kwargs", type=json.loads, default="{}")
    arg_parser.add_argument("--lrs", type=float, nargs="+", required=True)
    arg_parser.add_argument("--batch-size", type=int, required=True)
    arg_parser.add_argument("--epochs", type=int, required=True)
    arg_parser.add_argument("--dataset", type=str, required=True,
                            choices=get_subclass_names(Dataset))
    arg_parser.add_argument("--dataset-kwargs", type=json.loads, default="{}")
    arg_parser.add_argument("--model", type=str, required=True,
                            choices=get_subclass_names(Model))
    arg_parser.add_argument("--metrics", type=str, nargs="+")
    arg_parser.add_argument("--saved-loss", type=str, choices=["train", "test"],
                            default="train")
    arg_parser.add_argument("--save-path", type=str, required=True)
    args = arg_parser.parse_args()

    # Load data
    dataset = build_subclass_object(Dataset, args.dataset, args.dataset_kwargs)
    X_tr, y_tr = dataset.train_data
    X_te, y_te = dataset.test_data

    # For the callback, X and y depend on saved-loss
    X_cb, y_cb = (X_tr, y_tr) if args.saved_loss is "train" else (X_te, y_te)

    # Loop over different learning rates
    best_final_loss = None
    for lr in args.lrs:
        print("lr {}".format(lr))
        model = get_subclass_from_name(Model, args.model)(dataset)

        callbacks = [
            BatchLossHistory(),
            EpochFullLossHistory(X_cb, y_cb, args.batch_size)
        ]
        args.opt_kwargs["lr"] = lr
        if args.opt == "Eve":
            args.opt_kwargs["loss_min"] = model.loss_min
            callbacks.append(EveMonitor())
        opt = build_subclass_object(Optimizer, args.opt, args.opt_kwargs)

        # Compile and train
        model.model.compile(loss=model.loss, optimizer=opt,
                            metrics=args.metrics)

        model.model.fit(X_tr, y_tr, batch_size=args.batch_size,
                        epochs=args.epochs, callbacks=callbacks)

        full_losses = callbacks[1].losses

        if best_final_loss is None or min(full_losses) < best_final_loss:
            best_final_loss = min(full_losses)
            best_full_losses = full_losses
            best_batch_losses = callbacks[0].batch_losses
            best_lr = lr
            if args.opt == "Eve":
                best_eve_monitor = callbacks[2]
        print()

    # Save results
    save_data = {
        "cmd_args": args,
        "best_full_losses": best_full_losses,
        "best_batch_losses": best_batch_losses,
        "best_lr": best_lr
    }
    if args.opt == "Eve":
        best_eve_data = best_eve_monitor.get_data()
        for k, v in best_eve_data.items():
            save_data["best_{}".format(k)] = v
    save_pkl(save_data, args.save_path)


if __name__ == "__main__":
    main()
