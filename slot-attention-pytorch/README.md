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

## Hyperparameter Tuning with W&B Sweeps

This project is integrated with Weights & Biases (W&B) for automated hyperparameter tuning. The `sweep.yaml` file contains the configuration for a search over key hyperparameters like `learning_rate`, `num_slots`, and `batch_size`.

To find the best hyperparameters for the model, follow these steps:

**1. Log in to W&B**

From your terminal, log in to your W&B account.
```bash
wandb login
```

**2. Initialize the Sweep**

Navigate to the `slot-attention-pytorch` directory and start the sweep. W&B will print a unique SWEEP_ID that you will use in the next step.
```bash
cd slot-attention-pytorch
wandb sweep sweep.yaml
```

**3. Run the W&B Agent**

On your training machine (e.g., your Thunder Compute instance), run the W&B agent with the SWEEP_ID provided by the previous command. The agent will automatically run the training script with different hyperparameter combinations.
```bash
# Replace <USERNAME>/<PROJECT_NAME>/<SWEEP_ID> with the ID from the previous step
wandb agent <USERNAME>/<PROJECT_NAME>/<SWEEP_ID>
```

You can now monitor the progress and see all results, plots, and rankings on your W&B dashboard.

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
