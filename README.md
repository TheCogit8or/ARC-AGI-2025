# ARC-AGI-2025 Solver Attempt

This repository is an attempt to build a system capable of solving tasks from the Abstraction and Reasoning Corpus (ARC).

## Core Approach

The primary strategy involves using an object-centric learning model to decompose ARC grids into their constituent parts. This allows the system to reason about individual objects and their transformations, rather than dealing with the entire grid as a monolithic entity.

The core of this approach is implemented in the `slot-attention-pytorch` submodule, which contains a heavily modified version of the Slot Attention model adapted specifically for the discrete, grid-based nature of ARC tasks. For more details on the model architecture itself, please see the `README.md` inside that directory.

## End-to-End Workflow Guide

This guide covers the entire process, from setting up a cloud instance to training the model with Weights & Biases and evaluating the results.

### Part 1: Cloud Instance Setup & First-Time Configuration

This part describes how to set up the remote training environment. These steps only need to be done once.

1.  **Launch a Thunder Compute Instance:**
    Use the Thunder Compute CLI on your local machine to create and start a GPU instance. A `base` template with Ubuntu is recommended.
    ```bash
    # On your local machine
    tnr create
    tnr start 0
    ```

2.  **Connect and Clone Project:**
    Connect to your new instance and clone this repository. It is recommended to clone into the default home directory.
    ```bash
    # On your local machine
    tnr connect 0

    # Now on the remote server
    git clone https://github.com/your-repo/ARC-AGI-2025.git
    cd ARC-AGI-2025
    ```

3.  **Run the Setup Script:**
    This script prepares the environment by downloading the ARC dataset, creating a Python virtual environment (`arc_venv`), and installing all dependencies.
    ```bash
    # On the remote server
    chmod +x ./train_on_thunder.sh
    ./train_on_thunder.sh
    ```
    *Note: The script will fail on its final step (the test run), which is expected. Its main purpose is to set up the environment.*

### Part 2: Training with W&B Sweeps

This is the main workflow loop for running hyperparameter tuning experiments.

1.  **Connect and Sync:**
    If you are not already connected, connect to your instance. Always pull the latest code from your development branch to ensure your changes are synced.
    ```bash
    # On your local machine
    tnr connect 0

    # On the remote server
    cd ARC-AGI-2025
    git pull origin enhance-slot-attention-model
    ```

2.  **Activate Environment:**
    Activate the Python virtual environment in your terminal session.
    ```bash
    # On the remote server
    source arc_venv/bin/activate
    ```

3.  **Log in to W&B:**
    Ensure you are logged into the correct Weights & Biases account.
    ```bash
    # On the remote server
    wandb login --relogin
    ```

4.  **Start a New Sweep:**
    Navigate to the submodule directory and initialize a new sweep. This will generate a new Sweep ID.
    ```bash
    # On the remote server
    cd slot-attention-pytorch
    wandb sweep sweep.yaml
    ```

5.  **Run the W&B Agent:**
    Copy the `wandb agent <YOUR_SWEEP_COMMAND>` from the previous step's output and run it. The agent will now begin training models.
    ```bash
    # On the remote server
    wandb agent <YOUR_SWEEP_COMMAND>
    ```
    Let the agent run for at least one full trial (100 epochs), then stop it with `Ctrl+C`.

### Part 3: Evaluating a Trained Model

Once a model has been trained, follow these steps to visualize its segmentation performance.

1.  **Modify the Test Script:**
    The `arc_training_code.py` script is dual-purpose. To use it for testing, comment out the `train_arc_slot_attention(args)` call and uncomment the `test_segmentation(...)` call at the end of the file. Update the path to point to the model you want to test (saved in `slot-attention-pytorch/models/`).

2.  **Sync and Run Test:**
    Commit and push this change from your local machine, then pull it on the remote server. Then, run the script.
    ```bash
    # On the remote server (with venv active)
    python slot-attention-pytorch/arc_training_code.py
    ```
    This will generate a timestamped PNG file (e.g., `segmentation_20250629_150000.png`) in the project root.

3.  **Download and View Results:**
    Use `scp` on your **local machine** to download the image file for viewing. You can get your instance's IP address from `tnr status`.
    ```bash
    # On your local machine
    scp user@<INSTANCE_IP>:/path/to/your/project/segmentation_...png .
    ```

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







