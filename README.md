# eve

This repository contains code and results for the paper

**[Eve: A Gradient Based Optimization Method with Locally and Globally Adaptive Learning Rates](https://arxiv.org/abs/1611.01505)**<br>
[Hiroaki Hayashi](https://www.cs.cmu.edu/~hiroakih/)<sup>\*</sup>,
[Jayanth Koushik](https://jayanthkoushik.github.io)<sup>\*</sup>,
[Graham Neubig](http://phontron.com)<br>
(\* equal contribution)

## Setup
A conda environment to run experiments can be created by running

```bash
conda env create -f environment.yml
```

The environment is activated/deactivated using

```bash
source activate eve
source deactivate eve
```

A suitable Keras backend is required for GPU support. Refer to [the documentation](https://keras.io/#installation)
for instructions.

## Code
A Keras implementation of the algorithm is in `eve/optim/eve.py`. The `Eve`
class in this file can be passed to the `model.compile` method in Keras (using
the `optimizer` argument).

Scripts for the various experiments are inside `eve/exp/runners`. Run these
scripts from the root directory, as such:

```bash
python -m eve.exp.runners.compsched --help
```

The help command provides information about choices for the various arguments.
Learning rate schedules and datasets are referred to by their class names
(in `eve/exp/lrscheds.py` and `eve/exp/datasets.py` respectively). Arguments to
these classes are passed through the command line as json strings. Refer to
the paper for values used in our experiments.

## Citation
    @article{hayashi2017eve,
      title={Eve: A Gradient Based Optimization Method with Locally and Globally Adaptive Learning Rates},
      author={Hayashi, Hiroaki and Koushik, Jayanth and Neubig, Graham},
      journal={arXiv preprint arXiv:1611.01505},
      year={2017}
    }
