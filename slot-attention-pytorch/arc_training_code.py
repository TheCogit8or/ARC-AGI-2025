import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime
import argparse
import wandb

# Import our modified model
from modified_slot_attention import ARCSlotAttentionAutoEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ARCDataset(Dataset):
    """Dataset for ARC grids - keeps discrete values"""
    def __init__(self, json_dir, max_size=30):
        self.data = []
        self.max_size = max_size
        self._load_data(json_dir)

    def _load_data(self, json_dir):
        """Load all grids from JSON files"""
        for file_name in os.listdir(json_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(json_dir, file_name)
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                
                # Collect all grids (both input and output)
                for key in ["train", "test"]:
                    for sample in json_data.get(key, []):
                        # Add both input and output grids
                        input_grid = self._pad_grid(sample["input"])
                        output_grid = self._pad_grid(sample["output"])
                        
                        if input_grid is not None:
                            self.data.append(input_grid)
                        if output_grid is not None:
                            self.data.append(output_grid)

    def _pad_grid(self, grid):
        """Pad grid to max_size x max_size, keeping discrete values"""
        grid = np.array(grid)
        height, width = grid.shape
        
        # Skip grids that are too large
        if height > self.max_size or width > self.max_size:
            return None
            
        # Pad with zeros (background)
        padded = np.zeros((self.max_size, self.max_size), dtype=np.int32)
        padded[:height, :width] = grid
        
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ARCLoss(nn.Module):
    """Custom loss function for ARC discrete grids"""
    def __init__(self, reconstruction_weight=1.0, mask_sparsity_weight=0.1, foreground_weight=10.0, num_colors=10):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.mask_sparsity_weight = mask_sparsity_weight
        
        # Create a weight tensor for the loss function
        # foreground_weight > 1 gives more importance to object pixels
        class_weights = torch.ones(num_colors)
        class_weights[1:] = foreground_weight  # Weight for colors 1-9
        self.register_buffer('class_weights', class_weights)
        
    def forward(self, outputs, target_grid):
        # outputs: dict with 'combined_reconstruction', 'slot_masks', etc.
        # target_grid: [batch_size, height, width] with discrete values
        
        combined_recon = outputs['combined_reconstruction']  # [batch_size, height, width, num_colors]
        slot_masks = outputs['slot_masks']  # [batch_size, num_slots, height, width, 1]
        
        # Convert target to one-hot for comparison
        target_onehot = F.one_hot(target_grid, num_classes=combined_recon.shape[-1]).float()
        
        # Reconstruction loss (cross-entropy)
        recon_loss = F.cross_entropy(
            combined_recon.permute(0, 3, 1, 2),  # [batch_size, num_colors, height, width]
            target_grid,
            weight=self.class_weights  # Apply the class weights here
        )
        
        # Mask sparsity loss to encourage slots to specialize
        mask_entropy = -torch.sum(slot_masks * torch.log(slot_masks + 1e-8), dim=1)  # [batch_size, height, width, 1]
        sparsity_loss = torch.mean(mask_entropy)
        
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.mask_sparsity_weight * sparsity_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'sparsity_loss': sparsity_loss
        }

def visualize_segmentation(model, dataset, num_examples=3):
    """Visualize the segmentation results"""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 8, figsize=(20, num_examples * 3))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_examples):
            # Get a sample
            sample = dataset[i].unsqueeze(0).to(device)  # Add batch dimension
            
            # Get segmentation
            segments = model.get_individual_segments(sample, threshold=0.1)
            
            # Plot original
            axes[i, 0].imshow(sample[0].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Plot segments
            for j, segment in enumerate(segments[0][:7]):  # Up to 7 segments
                if j < 7:
                    axes[i, j+1].imshow(segment['grid'].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
                    axes[i, j+1].set_title(f'Slot {j+1}\n(conf: {segment["confidence"]:.2f})')
                    axes[i, j+1].axis('off')
            
            # Fill remaining slots with empty plots
            for j in range(len(segments[0]), 7):
                axes[i, j+1].axis('off')
    
    plt.tight_layout()
    # Save the figure instead of showing it, so we can view it outside the SSH session.
    plt.savefig("segmentation_results.png")
    plt.close(fig)
    print("Segmentation visualization saved to segmentation_results.png")

def train_arc_slot_attention(args):
    """Main training function"""
    
    # 1. Initialize W&B
    # anonymous="must" allows logging without an account
    run = wandb.init(anonymous="must", project="arc-slot-attention")
    
    # 2. Copy all hyperparameters from args and W&B config
    config = {
        **vars(args),
        **run.config,
    }
    
    # Create dataset
    # The path to the ARC data is now passed via arguments
    json_dir = args.data_dir
    if not os.path.isdir(json_dir):
        print(f"Error: Data directory not found at {json_dir}")
        return None
        
    dataset = ARCDataset(json_dir, max_size=config['grid_size'])
    print(f"Loaded {len(dataset)} grids from ARC dataset at {json_dir}")
    
    if len(dataset) == 0:
        print("Dataset is empty. Please check the data directory.")
        return None
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    model = ARCSlotAttentionAutoEncoder(
        resolution=(config['grid_size'], config['grid_size']),
        num_slots=config['num_slots'],
        num_iterations=config['num_iterations'],
        hid_dim=config['hid_dim'],
        num_colors=config['num_colors']
    ).to(device)
    
    # Create loss function and optimizer
    criterion = ARCLoss(
        reconstruction_weight=1.0,
        mask_sparsity_weight=0.1,
        foreground_weight=config['foreground_weight'],
        num_colors=config['num_colors']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    # 3. Log model architecture to W&B
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0}
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Update learning rate
            step = epoch * len(train_loader) + batch_idx
            lr = config['learning_rate']
            if step < config['warmup_steps']:
                lr = config['learning_rate'] * (step / config['warmup_steps'])
            else:
                lr = config['learning_rate'] * (config['decay_rate'] ** ((step - config['warmup_steps']) / config['decay_steps']))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            batch = batch.to(device)
            outputs = model(batch)
            losses = criterion(outputs, batch)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # Accumulate losses
            for key in ['total_loss', 'reconstruction_loss', 'sparsity_loss']:
                short_key = key.replace('_loss', '')
                train_losses[short_key] += losses[key].item()
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Log training losses to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_losses['total'],
            "train_recon_loss": train_losses['reconstruction'],
            "train_sparsity_loss": train_losses['sparsity'],
            "learning_rate": lr
        })
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                losses = criterion(outputs, batch)
                
                for key in ['total_loss', 'reconstruction_loss', 'sparsity_loss']:
                    short_key = key.replace('_loss', '')
                    val_losses[short_key] += losses[key].item()
        
        # Average validation losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # Log validation losses to W&B
        wandb.log({
            "val_loss": val_losses['total'],
            "val_recon_loss": val_losses['reconstruction'],
            "val_sparsity_loss": val_losses['sparsity']
        })
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {train_losses['total']:.4f} "
              f"(Recon: {train_losses['reconstruction']:.4f}, "
              f"Sparsity: {train_losses['sparsity']:.4f}) - "
              f"Val Loss: {val_losses['total']:.4f} - "
              f"Time: {datetime.timedelta(seconds=int(elapsed))}")
        
        # Visualize segmentation every 10 epochs and log to W&B
        if (epoch + 1) % 10 == 0:
            print("Visualizing segmentation results...")
            fig = visualize_segmentation(model, val_set, num_examples=3)
            wandb.log({"segmentation_examples": fig})
        
        # Save model checkpoint artifact to W&B every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Save a local checkpoint as well
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"./models/arc_slot_attention_epoch_{epoch+1}_{timestamp}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, save_path)
            print(f"Local model checkpoint saved to {save_path}")

            # Create and log a W&B artifact
            artifact = wandb.Artifact(f'arc-model-epoch-{epoch+1}', type='model')
            artifact.add_file(save_path)
            run.log_artifact(artifact)
            print("Model artifact saved to W&B")
    
    print("Training completed!")
    # Mark the run as finished
    run.finish()
    return model

def test_segmentation(model_path, dataset_path, num_examples=5):
    """Test the trained model on segmentation"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = ARCSlotAttentionAutoEncoder(
        resolution=(config['grid_size'], config['grid_size']),
        num_slots=config['num_slots'],
        num_iterations=config['num_iterations'],
        hid_dim=config['hid_dim'],
        num_colors=config['num_colors']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test dataset
    test_dataset = ARCDataset(dataset_path, max_size=config['grid_size'])
    
    # Visualize results
    visualize_segmentation(model, test_dataset, num_examples=num_examples)
    
    # Extract segments for analysis
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(test_dataset))):
            sample = test_dataset[i].unsqueeze(0).to(device)
            segments = model.get_individual_segments(sample, threshold=0.1)
            
            print(f"\nExample {i+1}:")
            print(f"Original grid shape: {sample.shape}")
            print(f"Number of segments found: {len(segments[0])}")
            
            for j, segment in enumerate(segments[0]):
                non_zero_positions = torch.nonzero(segment['grid'])
                print(f"  Segment {j+1}: {len(non_zero_positions)} non-zero pixels, "
                      f"confidence: {segment['confidence']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARC Slot Attention Model")
    
    # --- Data and Run Management ---
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory containing ARC JSON training files.")
    parser.add_argument('--grid_size', type=int, default=30,
                        help="The size to pad all grids to (grid_size x grid_size).")
    
    # --- Training Hyperparameters ---
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.0004,
                        help="Initial learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help="Number of warmup steps for learning rate.")
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help="Learning rate decay rate.")
    parser.add_argument('--decay_steps', type=int, default=100000,
                        help="Learning rate decay steps.")
    
    # --- Model Hyperparameters ---
    parser.add_argument('--num_slots', type=int, default=8,
                        help="Number of slots in the Slot Attention module.")
    parser.add_argument('--num_iterations', type=int, default=3,
                        help="Number of attention iterations.")
    parser.add_argument('--hid_dim', type=int, default=128,
                        help="Hidden dimension size in the model.")
    parser.add_argument('--num_colors', type=int, default=10,
                        help="Number of possible colors in ARC grids (0-9).")
    parser.add_argument('--foreground_weight', type=float, default=10.0,
                        help="Weight for foreground pixels in the loss function.")


    args = parser.parse_args()
    
    # Train the model
    trained_model = train_arc_slot_attention(args)