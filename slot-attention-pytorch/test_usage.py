"""
Example usage of the modified Slot Attention for ARC grid segmentation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from modified_slot_attention import ARCSlotAttentionAutoEncoder

def create_sample_arc_grid():
    """Create a simple synthetic ARC-like grid for testing"""
    grid = np.zeros((15, 15), dtype=np.int32)
    
    # Add some colored rectangles
    grid[2:5, 2:5] = 1  # Red square
    grid[8:11, 8:11] = 2  # Green square
    grid[2:5, 10:13] = 3  # Blue square
    grid[10:13, 2:5] = 4  # Yellow square
    
    # Add a line
    grid[7, 2:13] = 5  # Magenta line
    
    return torch.tensor(grid, dtype=torch.long)

def demonstrate_segmentation():
    """Demonstrate the segmentation capability"""
    
    # Create model
    model = ARCSlotAttentionAutoEncoder(
        resolution=(15, 15),
        num_slots=7,
        num_iterations=3,
        hid_dim=64,
        num_colors=10
    )
    
    # Create sample data
    sample_grid = create_sample_arc_grid()
    batch = sample_grid.unsqueeze(0)  # Add batch dimension
    
    print("Original grid shape:", batch.shape)
    print("Grid content:")
    print(batch[0].numpy())
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        segments = model.get_individual_segments(batch, threshold=0.01)
    
    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot original
    axes[0, 0].imshow(batch[0].numpy(), cmap='tab10', vmin=0, vmax=9)
    axes[0, 0].set_title('Original Grid')
    axes[0, 0].axis('off')
    
    # Plot reconstructed
    recon = torch.argmax(outputs['combined_reconstruction'][0], dim=-1)
    axes[0, 1].imshow(recon.numpy(), cmap='tab10', vmin=0, vmax=9)
    axes[0, 1].set_title('Reconstructed Grid')
    axes[0, 1].axis('off')
    
    # Plot individual segments
    for i, segment in enumerate(segments[0][:6]):
        row = i // 3
        col = (i % 3) + 2 if row == 0 else (i % 3) - 1
        if row == 1 and col < 0:
            continue
            
        axes[row, col].imshow(segment['grid'].numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[row, col].set_title(f'Segment {i+1}\nConf: {segment["confidence"]:.3f}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(segments[0]), 6):
        row = 1 if i >= 3 else 0
        col = (i % 3) + 2 if row == 0 else (i % 3) + 1
        if row < 2 and col < 4:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print segment analysis
    print(f"\nSegmentation Analysis:")
    print(f"Number of segments found: {len(segments[0])}")
    
    for i, segment in enumerate(segments[0]):
        non_zero_mask = segment['grid'] > 0
        unique_colors = torch.unique(segment['grid'][non_zero_mask])
        num_pixels = torch.sum(non_zero_mask).item()
        
        print(f"Segment {i+1}:")
        print(f"  - Confidence: {segment['confidence']:.3f}")
        print(f"  - Non-zero pixels: {num_pixels}")
        print(f"  - Colors used: {unique_colors.tolist()}")
        
        if num_pixels > 0:
            # Find bounding box
            coords = torch.nonzero(non_zero_mask)
            min_row, max_row = coords[:, 0].min().item(), coords[:, 0].max().item()
            min_col, max_col = coords[:, 1].min().item(), coords[:, 1].max().item()
            print(f"  - Bounding box: ({min_row}, {min_col}) to ({max_row}, {max_col})")

def analyze_arc_file(json_path, model_path=None):
    """Analyze a real ARC file using the trained model"""
    import json
    from arc_training_code import ARCDataset
    
    # Load ARC file
    with open(json_path, 'r') as f:
        arc_data = json.load(f)
    
    print(f"Analyzing ARC file: {json_path}")
    print(f"Number of training examples: {len(arc_data.get('train', []))}")
    print(f"Number of test examples: {len(arc_data.get('test', []))}")
    
    # Create model
    if model_path and os.path.exists(model_path):
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        model = ARCSlotAttentionAutoEncoder(
            resolution=(config['grid_size'], config['grid_size']),
            num_slots=config['num_slots'],
            num_iterations=config['num_iterations'],
            hid_dim=config['hid_dim'],
            num_colors=config['num_colors']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model")
    else:
        # Use untrained model for demonstration
        model = ARCSlotAttentionAutoEncoder(
            resolution=(30, 30),
            num_slots=7,
            num_iterations=3,
            hid_dim=64,
            num_colors=10
        )
        print("Using untrained model (for demonstration)")
    
    model.eval()
    
    # Analyze first training example
    if arc_data.get('train'):
        example = arc_data['train'][0]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"\nFirst training example:")
        print(f"Input grid shape: {input_grid.shape}")
        print(f"Output grid shape: {output_grid.shape}")
        
        # Pad grids to model size
        def pad_to_size(grid, target_size=30):
            h, w = grid.shape
            if h > target_size or w > target_size:
                return None
            padded = np.zeros((target_size, target_size), dtype=np.int32)
            padded[:h, :w] = grid
            return torch.tensor(padded, dtype=torch.long)
        
        input_tensor = pad_to_size(input_grid)
        output_tensor = pad_to_size(output_grid)
        
        if input_tensor is not None and output_tensor is not None:
            # Analyze input segmentation
            print("\n--- Input Grid Segmentation ---")
            with torch.no_grad():
                input_batch = input_tensor.unsqueeze(0)
                input_segments = model.get_individual_segments(input_batch, threshold=0.05)
                
                for i, segment in enumerate(input_segments[0]):
                    non_zero_mask = segment['grid'] > 0
                    num_pixels = torch.sum(non_zero_mask).item()
                    if num_pixels > 0:
                        unique_colors = torch.unique(segment['grid'][non_zero_mask])
                        print(f"Input Segment {i+1}: {num_pixels} pixels, colors {unique_colors.tolist()}, conf: {segment['confidence']:.3f}")
            
            # Analyze output segmentation
            print("\n--- Output Grid Segmentation ---")
            with torch.no_grad():
                output_batch = output_tensor.unsqueeze(0)
                output_segments = model.get_individual_segments(output_batch, threshold=0.05)
                
                for i, segment in enumerate(output_segments[0]):
                    non_zero_mask = segment['grid'] > 0
                    num_pixels = torch.sum(non_zero_mask).item()
                    if num_pixels > 0:
                        unique_colors = torch.unique(segment['grid'][non_zero_mask])
                        print(f"Output Segment {i+1}: {num_pixels} pixels, colors {unique_colors.tolist()}, conf: {segment['confidence']:.3f}")
            
            # Visualize both
            fig, axes = plt.subplots(2, 8, figsize=(20, 6))
            
            # Input row
            axes[0, 0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
            axes[0, 0].set_title('Input Grid')
            axes[0, 0].axis('off')
            
            for i, segment in enumerate(input_segments[0][:7]):
                axes[0, i+1].imshow(segment['grid'][:input_grid.shape[0], :input_grid.shape[1]].numpy(), 
                                   cmap='tab10', vmin=0, vmax=9)
                axes[0, i+1].set_title(f'In-Seg {i+1}')
                axes[0, i+1].axis('off')
            
            # Output row
            axes[1, 0].imshow(output_grid, cmap='tab10', vmin=0, vmax=9)
            axes[1, 0].set_title('Output Grid')
            axes[1, 0].axis('off')
            
            for i, segment in enumerate(output_segments[0][:7]):
                axes[1, i+1].imshow(segment['grid'][:output_grid.shape[0], :output_grid.shape[1]].numpy(), 
                                   cmap='tab10', vmin=0, vmax=9)
                axes[1, i+1].set_title(f'Out-Seg {i+1}')
                axes[1, i+1].axis('off')
            
            plt.tight_layout()
            plt.show()

def comprehensive_test():
    """Run comprehensive tests of the system"""
    print("=== Comprehensive Test of Modified Slot Attention ===\n")
    
    # Test 1: Basic functionality with synthetic data
    print("Test 1: Basic segmentation with synthetic data")
    try:
        demonstrate_segmentation()
        print("✓ Basic segmentation test passed\n")
    except Exception as e:
        print(f"✗ Basic segmentation test failed: {e}\n")
    
    # Test 2: Model architecture
    print("Test 2: Model architecture validation")
    try:
        model = ARCSlotAttentionAutoEncoder(
            resolution=(20, 20),
            num_slots=5,
            num_iterations=2,
            hid_dim=32,
            num_colors=10
        )
        
        # Test forward pass
        sample_input = torch.randint(0, 10, (2, 20, 20))  # Batch of 2
        outputs = model(sample_input)
        
        print(f"✓ Model forward pass successful")
        print(f"  - Input shape: {sample_input.shape}")
        print(f"  - Combined reconstruction shape: {outputs['combined_reconstruction'].shape}")
        print(f"  - Slot reconstructions shape: {outputs['slot_reconstructions'].shape}")
        print(f"  - Slot masks shape: {outputs['slot_masks'].shape}")
        
        # Test segmentation extraction
        segments = model.get_individual_segments(sample_input, threshold=0.01)
        print(f"  - Number of segments for batch: {[len(seg) for seg in segments]}")
        print("✓ Model architecture test passed\n")
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}\n")
    
    # Test 3: Loss function
    print("Test 3: Loss function validation")
    try:
        from arc_training_code import ARCLoss
        
        criterion = ARCLoss()
        sample_input = torch.randint(0, 10, (3, 15, 15))
        model_outputs = model(sample_input)
        
        losses = criterion(model_outputs, sample_input)
        print(f"✓ Loss computation successful")
        print(f"  - Total loss: {losses['total_loss'].item():.4f}")
        print(f"  - Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
        print(f"  - Sparsity loss: {losses['sparsity_loss'].item():.4f}")
        print("✓ Loss function test passed\n")
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}\n")

if __name__ == "__main__":
    # Run basic demonstration
    print("Running basic segmentation demonstration...")
    demonstrate_segmentation()
    
    # Run comprehensive tests
    print("\n" + "="*50)
    comprehensive_test()
    
    # Example of how to analyze a real ARC file (uncomment to use)
    # analyze_arc_file("./ARC-AGI-2/data/training/00d62c1b.json")