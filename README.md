# eve

This repository contains code and results for the paper

**Eve: A Gradient Based Optimization Method with Locally and Globally Adaptive
Learning rate**<br>
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

Note that this environment is not sufficient to generate figures;
that requires a latex installation. For figures, the provided docker file could
be more convenient. It does not support GPUs however, and is not suitable for
running experiments except for debugging. Refer to the Docker docks for details
on using images. The gist is that the image is created by running

```bash
docker build -t <image name>
```

at the root of the repository. Then launch a container with the image using

```bash
docker run --rm -it -v <repository root dir>:/project
```

This will put you in a Ubuntu environment where `/project` is the root of the
repository.
