# Modified Slot Attention for ARC

This repository contains a modified implementation of "Object-Centric Learning with Slot Attention" adapted specifically for the Abstraction and Reasoning Corpus (ARC).

The core model has been re-engineered to work directly with the discrete, grid-based data format of ARC tasks, rather than traditional image data. The goal of this model is to perform unsupervised object segmentation on ARC grids.

## Key Modifications

- **`modified_slot_attention.py`**: The original CNN-based encoder/decoder has been replaced with an `ARCEncoder` and `ARCDecoder`.
    - The `ARCEncoder` uses an `nn.Embedding` layer to process the discrete integer values (colors) of the ARC grids.
    - The `ARCDecoder` uses transposed convolutions to reconstruct the segmented objects spatially.
- **`arc_training_code.py`**: This is the main script for training the model. It includes:
    - An `ARCDataset` class to load, pad, and batch ARC task files from JSON format.
    - A complete training and validation loop with learning rate scheduling, logging, and model checkpointing.
    - Visualization functions to inspect segmentation results.
- **`test_usage.py`**: An example script demonstrating how to load a trained model and use it to segment a sample ARC grid.
- **`requirements.txt`**: Contains the necessary Python packages to run the code.

The original `dataset.py` and `eval.ipynb` have been removed as their functionality is now covered by the ARC-specific scripts.

## How to Train

The entire training process is managed by the `train_on_thunder.sh` script located in the project's root directory. This script is designed to be run on a fresh cloud instance (e.g., on [Thunder Compute](https://www.thundercompute.com/)) and handles all setup automatically.

To start training, navigate to the project root and run:
```bash
# First, make the script executable
chmod +x ../train_on_thunder.sh

# Run the script
../train_on_thunder.sh
```

The script will:
1.  Clone the `fchollet/ARC-AGI` dataset from GitHub into a directory named `arc_dataset`.
2.  Create a local Python virtual environment (`arc_venv`).
3.  Install all required packages from `requirements.txt`.
4.  Launch `arc_training_code.py` with the appropriate data directory and hyperparameters.

The script is the single, recommended entry point for training the model.

## Citation

```bibtex
@misc{locatello2020objectcentric,
    title = {Object-Centric Learning with Slot Attention},
    author = {Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
    year = {2020},
    eprint = {2006.15055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
