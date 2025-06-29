#!/bin/bash

# =================================================================================
# Master Script for Training ARC Slot Attention on a Cloud Instance
#
# This script automates the entire process of:
# 1. Cloning the required ARC dataset from GitHub.
# 2. Setting up a dedicated Python virtual environment.
# 3. Installing all necessary dependencies.
# 4. Running the training script with the correct parameters.
#
# Usage:
#   - Make sure this script is executable: chmod +x train_on_thunder.sh
#   - Run the script: ./train_on_thunder.sh
# =================================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# GitHub repository for the original ARC-AGI dataset
DATASET_REPO="https://github.com/fchollet/ARC-AGI.git"
# Directory to clone the dataset into
DATASET_DIR="arc_dataset"
# Path to the training data within the cloned repository
TRAINING_DATA_PATH="$DATASET_DIR/data/training"
# Python virtual environment directory
VENV_DIR="arc_venv"
# Path to the requirements file
REQUIREMENTS_FILE="slot-attention-pytorch/requirements.txt"
# Path to the main Python training script
TRAINING_SCRIPT="slot-attention-pytorch/arc_training_code.py"

echo "--- Starting ARC Slot Attention Training Setup ---"

# 1. Clone ARC Dataset
if [ -d "$DATASET_DIR" ]; then
    echo "Dataset directory '$DATASET_DIR' already exists. Skipping clone."
else
    echo "Cloning ARC dataset from $DATASET_REPO..."
    git clone $DATASET_REPO $DATASET_DIR
fi

# 2. Create and Activate Python Virtual Environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Activating."
else
    echo "Creating Python virtual environment in '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# 3. Install Dependencies
echo "Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

# 4. Run Training
echo "Starting the training process..."
echo "Dataset location: $TRAINING_DATA_PATH"

python3 $TRAINING_SCRIPT \
    --data_dir "$TRAINING_DATA_PATH" \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0004 \
    --num_slots 8 \
    --grid_size 30

echo "--- Training script finished. ---"
deactivate
echo "Virtual environment deactivated. Setup complete." 