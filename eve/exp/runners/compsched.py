"""compsched.py: runner for comparing different lr schedules with Eve.

This file compares Adam+different learning rate schedules with Eve on
minimizing the training loss of models.
"""

import json
import itertools
from argparse import ArgumentParser

import numpy as np
np.random.seed(1862569059)

from keras.optimizers import Adam

from eve.optim.eve import Eve
from eve.exp.models import Model
from eve.exp.datasets import Dataset
from eve.exp.lrscheds import LRSchedule
from eve.exp.callbacks import EpochFullLossHistory
from eve.exp.utils import (build_subclass_object, get_subclass_names,
                           get_subclass_from_name, save_pkl)


def main():
    """Run experiment for comparing learning rate schedules."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--lr-schedule", type=str, required=True,
                            choices=get_subclass_names(LRSchedule)+["Eve"])
    arg_parser.add_argument("--lrs", type=float, nargs="+", required=True)
    arg_parser.add_argument("--decays", type=float, nargs="+")
    arg_parser.add_argument("--batch-size", type=int, required=True)
    arg_parser.add_argument("--epochs", type=int, required=True)
    arg_parser.add_argument("--dataset", type=str, required=True,
                            choices=get_subclass_names(Dataset))
    arg_parser.add_argument("--dataset-kwargs", type=json.loads, default="{}")
    arg_parser.add_argument("--model", type=str, required=True,
                            choices=get_subclass_names(Model))
    arg_parser.add_argument("--save-path", type=str, required=True)
    args = arg_parser.parse_args()

    # Load data
    dataset = build_subclass_object(Dataset, args.dataset, args.dataset_kwargs)
    X, y = dataset.train_data

    if args.lr_schedule == "Eve":
        # No decay parameter to loop over.
        # A dummy value is needed in the args.decays list so that
        # itertools.product works.
        args.decays = [-1]

    # Loop over different learning rates and decays
    best_final_loss = None
    for lr, decay in itertools.product(args.lrs, args.decays):
        if args.lr_schedule == "Eve":
            print("lr {}".format(lr))
        else:
            print("lr {}, decay {}".format(lr, decay))

        model = get_subclass_from_name(Model, args.model)(dataset)
        callbacks = [EpochFullLossHistory(X, y, args.batch_size)]

        # Set up optimizer.
        # If learning rate schedule has the special value "Eve",
        # then it's a normal optimizer. Otherwise, a lr schedule
        # is added to the callback list.
        if args.lr_schedule == "Eve":
            opt = Eve(lr=lr, loss_min=model.loss_min)
        else:
            opt = Adam(lr=lr)
            lr_schedule = get_subclass_from_name(LRSchedule, args.lr_schedule)
            callbacks.append(lr_schedule(lr, decay))

        # Compile and train
        model.model.compile(loss=model.loss, optimizer=opt)
        model.model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs,
                        callbacks=callbacks)
        full_losses = callbacks[0].losses

        if best_final_loss is None or full_losses[-1] < best_final_loss:
            best_final_loss = full_losses[-1]
            best_full_losses = full_losses
            best_lr = lr
            best_decay = decay
        print()

    # Save results
    save_data = {
        "cmd_args": args,
        "best_full_losses": best_full_losses,
        "best_lr": best_lr,
        "best_decay": best_decay
    }
    save_pkl(save_data, args.save_path)


if __name__ == "__main__":
    main()
