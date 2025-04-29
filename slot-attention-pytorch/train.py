import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch


import datetime

import random


import matplotlib.pyplot as plt
import numpy as np


from torch.utils.data import random_split



import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, Pad





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""
parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

opt = parser.parse_args()

"""

if not os.path.exists('./Models'):
    os.makedirs('./Models')

class Options:
    def __init__(self):
        self.model_dir = './Models'
        self.seed = 0
        self.batch_size = 16
        self.num_slots = 7
        self.num_iterations = 3
        self.hid_dim = 64
        self.learning_rate = 0.0004
        self.warmup_steps = 10000
        self.decay_rate = 0.5
        self.decay_steps = 100000
        self.num_workers = 4
        self.num_epochs = 10

opt = Options()

resolution = (64, 64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#train_set = CLEVR('train')



# Define a color map (integer values to RGB colors)
COLOR_MAP = {
    0: [0, 0, 0],       # Black
    1: [255, 0, 0],     # Red
    2: [0, 255, 0],     # Green
    3: [0, 0, 255],     # Blue
    4: [255, 255, 0],   # Yellow
    5: [255, 0, 255],   # Magenta
    6: [0, 255, 255],   # Cyan
    7: [128, 128, 128], # Gray
    8: [255, 165, 0],   # Orange
    9: [128, 0, 128],   # Purple
}





class JSONImageDataset(Dataset):
    def __init__(self, json_dir, resolution=(128, 128)):
        """
        Args:
            json_dir (str): Path to the directory containing .json files.
            resolution (tuple): Target resolution for scaling and padding (height, width).
        """
        self.data = []
        self.resolution = resolution
        self._load_data(json_dir)

    def _load_data(self, json_dir):
        """
        Load all grids (input and output) from the .json files in the directory.
        """
        for file_name in os.listdir(json_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(json_dir, file_name)
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                
                # Collect all grids (both "train" and "test", "input" and "output")
                for key in ["train", "test"]:
                    for sample in json_data.get(key, []):
                        # Add both input and output grids to the dataset
                        self.data.append(self._scale_and_pad(torch.tensor(sample["input"], dtype=torch.float32).unsqueeze(0)))
                        self.data.append(self._scale_and_pad(torch.tensor(sample["output"], dtype=torch.float32).unsqueeze(0)))

    def _scale_and_pad(self, tensor):
        """
        Scales, pads, and converts a tensor to RGB format using the COLOR_MAP.
        Args:
            tensor (torch.Tensor): Input tensor of shape [1, height, width].
        Returns:
            torch.Tensor: Scaled, padded, and RGB-converted tensor of shape [3, target_height, target_width].
        """
        _, height, width = tensor.shape

        # Calculate scaling factor
        scale_factor = min(self.resolution[0] / height, self.resolution[1] / width)
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)

        # Resize the tensor using nearest neighbor interpolation
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0)

        # Apply COLOR_MAP using torch indexing
        color_map_tensor = torch.tensor([COLOR_MAP[i] for i in range(len(COLOR_MAP))], dtype=torch.float32) / 255.0

        # Map each pixel value to its RGB color
        rgb_image = color_map_tensor[tensor.long()]  # Shape: [height, width, 3] or [1, height, width, 3]

        # Remove any extra batch dimension if it exists
        rgb_image = rgb_image.squeeze(0)  # Ensure shape is [height, width, 3]

        # Rearrange dimensions from [height, width, 3] (HWC) to [3, height, width] (CHW)
        rgb_image = rgb_image.permute(2, 0, 1)  # Shape: [3, height, width]

        # Calculate padding
        pad_h = self.resolution[0] - new_height
        pad_w = self.resolution[1] - new_width
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)

        # Pad the RGB tensor
        return F.pad(rgb_image, padding)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# Directory containing the .json files

#json_dir = r"c:\Users\todth\Documents\GitHub\ARC-AGI-2025\ARC-AGI-2\data\training"

# Find the parent folder "ARC-AGI-2025"
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Construct the dynamic path starting from "ARC-AGI-2025"
json_dir = os.path.join(parent_folder, "ARC-AGI-2", "data", "training")

print(f"JSON Directory: {json_dir}")



# Create the dataset and DataLoader
dataset = JSONImageDataset(json_dir, resolution)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Print the total number of images in the dataset
print(f"Total number of images in the dataset: {len(dataset)}")






def plot_json_image(image, title=""):
    """
    Plots a single JSON image in color.
    
    Args:
        image (torch.Tensor or numpy.ndarray): 3D RGB tensor in [3, height, width] format.
        title (str): Title of the plot.
    """
    # Convert from [3, height, width] (CHW) to [height, width, 3] (HWC) for plotting
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC and numpy array
    elif isinstance(image, np.ndarray):
        image = np.transpose(image, (1, 2, 0))  # Convert to HWC if it's a numpy array

    # Ensure the image is in uint8 format for plotting
    image = (image * 255).astype(np.uint8)

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Display a random grid
sample_image = dataset[random.randint(30, 40)]  # Get a sample from the dataset
plot_json_image(sample_image, title="Sample from Dataset")






model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]



# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_set, val_set = random_split(dataset, [train_size, val_size])



train_dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=0)


"""


# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
val_dataloader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
"""
# Print dataset sizes
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")














optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0
for epoch in range(opt.num_epochs):
    model.train()
    total_loss = 0

    # Training loop
    for sample in tqdm(train_dataloader):
        i += 1

        # Update learning rate
        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))
        optimizer.param_groups[0]['lr'] = learning_rate

        # Forward pass
        image = sample.to(device)  # Fix: Use sample directly
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        # Debug during the first batch
        if epoch == 0 and i == 1:
            print(f"Image shape: {image.shape}")
            print(f"Recon_combined shape: {recon_combined.shape}")
            print(f"Loss: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clean up
        del recons, masks, slots

    total_loss /= len(train_dataloader)
    print(f"Epoch: {epoch}, Loss: {total_loss}, Time: {datetime.timedelta(seconds=time.time() - start)}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sample in val_dataloader:
            image = sample.to(device)  # Fix: Use sample directly
            recon_combined, recons, masks, slots = model(image)
            loss = criterion(recon_combined, image)
            total_val_loss += loss.item()

    total_val_loss /= len(val_dataloader)
    print(f"Validation Loss: {total_val_loss}")

    # Save the model every 10 epochs
    if epoch % 10 == 0:
        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(
            {'model_state_dict': model.state_dict()},
            f"{opt.model_dir}/model_{datetime_string}.pth"
        )






def visualize_slots(image, recons, masks, epoch):
    """
    Visualizes the input image, reconstructed slots, and attention masks.
    
    Args:
        image (torch.Tensor): The input image of shape [batch_size, 3, height, width].
        recons (torch.Tensor): Reconstructed slots of shape [batch_size, num_slots, 3, height, width].
        masks (torch.Tensor): Attention masks of shape [batch_size, num_slots, 1, height, width].
        epoch (int): Current epoch number (for display purposes).
    """
    batch_size, num_slots, _, height, width = recons.shape

    # Visualize the first image in the batch
    input_image = image[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    slot_recons = recons[0]  # Shape: [num_slots, 3, height, width]
    slot_masks = masks[0]  # Shape: [num_slots, 1, height, width]

    # Plot the input image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_slots + 1, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis("off")

    # Plot each slot's reconstruction and mask
    for slot_idx in range(num_slots):
        # Reconstruction
        recon_image = slot_recons[slot_idx].permute(1, 2, 0).cpu().numpy()  # Convert to HWC
        plt.subplot(2, num_slots + 1, slot_idx + 2)
        plt.imshow(recon_image)
        plt.title(f"Slot {slot_idx} Recon")
        plt.axis("off")

        # Mask
        mask_image = slot_masks[slot_idx, 0].cpu().numpy()  # Convert to 2D
        plt.subplot(2, num_slots + 1, num_slots + slot_idx + 2)
        plt.imshow(mask_image, cmap="gray")
        plt.title(f"Slot {slot_idx} Mask")
        plt.axis("off")

    plt.suptitle(f"Epoch {epoch} - Slot Reconstructions and Masks", fontsize=16)
    plt.tight_layout()
    plt.show()







# Example: Add this to your training loop
if epoch % 10 == 0:  # Visualize every 10 epochs
    with torch.no_grad():
        for sample in val_dataloader:
            image = sample.to(device)
            _, recons, masks, _ = model(image)

            # Visualize the first batch
            visualize_slots(image, recons, masks, epoch)
            break  # Only visualize one batch




# TODO: Turn workers > 0 for multiprocessing if that wasn't causing issues