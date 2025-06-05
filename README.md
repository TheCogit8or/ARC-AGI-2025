# ARC-AGI-2025 Solver Attempt

This repository is an attempt to build a system capable of solving tasks from the Abstraction and Reasoning Corpus (ARC).

## Core Approach

The primary strategy involves using an object-centric learning model to decompose ARC grids into their constituent parts. This allows the system to reason about individual objects and their transformations, rather than dealing with the entire grid as a monolithic entity.

The core of this approach is implemented in the `slot-attention-pytorch` submodule, which contains a heavily modified version of the Slot Attention model adapted specifically for the discrete, grid-based nature of ARC tasks.

## Getting Started: Training the Segmentation Model

The entire process for training the object segmentation model is automated via a single script. This script is designed to be run on a cloud GPU instance (e.g., [Thunder Compute](https://www.thundercompute.com/)).

**To begin training, follow these steps:**

1.  Launch a cloud GPU instance with a standard Ubuntu environment.
2.  Sync this repository to the instance.
3.  From the project's root directory, run the master training script:

```bash
# First, make the script executable
chmod +x ./train_on_thunder.sh

# Run the script to start the full process
./train_on_thunder.sh
```

This script will automatically:
- Download the ARC dataset from GitHub.
- Set up a Python virtual environment.
- Install all required dependencies.
- Begin the training process.

For more detailed information on the model architecture, hyperparameter tuning, and advanced usage, please refer to the `README.md` file inside the `slot-attention-pytorch` directory.

# Citations


[1]

Yining Hong  1 (424) 832 6189
University of California, Los Angeles
Department of Computer Science
9401 Beolter Hall
E-mail: yninghong@gmail.com

https://github.com/evelinehong/slot-attention-pytorch.git

@misc{locatello2020objectcentric,
    title = {Object-Centric Learning with Slot Attention},
    author = {Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
    year = {2020},
    eprint = {2006.15055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}







