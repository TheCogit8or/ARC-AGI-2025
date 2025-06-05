import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SlotAttention(nn.Module):
    """Original SlotAttention module - keeping this unchanged"""
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

def build_grid(resolution):
    """Build coordinate grid for positional encoding"""
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class ARCEncoder(nn.Module):
    """Modified encoder for ARC discrete grids"""
    def __init__(self, resolution, hid_dim, num_colors=10):
        super().__init__()
        self.num_colors = num_colors
        
        # Embedding layer for discrete values
        self.color_embedding = nn.Embedding(num_colors, hid_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
        
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        # x shape: [batch_size, height, width] with discrete values
        batch_size, height, width = x.shape
        
        # Embed discrete values
        x = self.color_embedding(x.long())  # [batch_size, height, width, hid_dim]
        x = x.permute(0, 3, 1, 2)  # [batch_size, hid_dim, height, width]
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Rearrange for positional encoding
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, hid_dim]
        x = self.encoder_pos(x)
        
        # Flatten spatial dimensions
        x = torch.flatten(x, 1, 2)  # [batch_size, height*width, hid_dim]
        return x

class ARCDecoder(nn.Module):
    """Modified decoder that uses transposed convolutions for spatial reconstruction"""
    def __init__(self, hid_dim, resolution, num_colors=10):
        super().__init__()
        self.resolution = resolution
        self.num_colors = num_colors
        self.hid_dim = hid_dim
        
        # Initial grid size for broadcasting the slot vector
        self.decoder_initial_size = (8, 8)
        
        # Layers to project slot to initial grid
        self.fc = nn.Linear(hid_dim, self.decoder_initial_size[0] * self.decoder_initial_size[1] * hid_dim)
        
        # Positional embedding for the small initial grid
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        
        # Transposed convolutional layers for upsampling
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 3, stride=(1, 1), padding=1)
        
        # Final layer to produce color and mask predictions
        # Output channels = num_colors (for value distribution) + 1 (for the mask)
        self.conv4 = nn.ConvTranspose2d(hid_dim, num_colors + 1, 3, stride=(1, 1), padding=1)

    def forward(self, slots):
        # slots shape: [batch_size * num_slots, slot_dim]
        
        # Project and reshape slot vector to initial spatial grid
        x = self.fc(slots)
        x = x.reshape(-1, self.hid_dim, self.decoder_initial_size[0], self.decoder_initial_size[1])
        
        # Apply positional encoding
        x = self.decoder_pos(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Upsample using transposed convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x) # No activation, as we'll use softmax/sigmoid later
        
        # Crop to final resolution
        # The output of conv2 is 32x32, so we crop to the target resolution
        x = x[:, :, :self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1) # [batch_size*num_slots, height, width, num_colors+1]
        
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class ARCSlotAttentionAutoEncoder(nn.Module):
    """Modified Slot Attention model for ARC grids"""
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, num_colors=10):
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_colors = num_colors

        self.encoder = ARCEncoder(resolution, hid_dim, num_colors)
        self.decoder = ARCDecoder(hid_dim, resolution, num_colors)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=hid_dim,
            iters=num_iterations,
            eps=1e-8, 
            hidden_dim=128
        )

    def forward(self, grid):
        # grid shape: [batch_size, height, width] with discrete values
        batch_size = grid.shape[0]
        
        # Encode grid
        x = self.encoder(grid)  # [batch_size, height*width, hid_dim]
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Apply slot attention
        slots = self.slot_attention(x)  # [batch_size, num_slots, hid_dim]
        
        # Reshape for decoder
        slots_flat = slots.reshape(-1, slots.shape[-1])  # [batch_size*num_slots, hid_dim]
        
        # Decode slots
        decoded_output = self.decoder(slots_flat) # [batch_size*num_slots, height, width, num_colors+1]
        
        # Split decoded output into values and masks
        predicted_values, predicted_masks = torch.split(decoded_output, [self.num_colors, 1], dim=-1)
        
        # Reshape back to per-slot format
        predicted_values = predicted_values.reshape(batch_size, self.num_slots, *self.resolution, self.num_colors)
        predicted_masks = predicted_masks.reshape(batch_size, self.num_slots, *self.resolution, 1)
        
        # Apply softmax to get probability distributions over colors
        predicted_values = F.softmax(predicted_values, dim=-1)
        
        # Normalize masks across slots
        predicted_masks = F.softmax(predicted_masks, dim=1)
        
        # Combine predictions weighted by masks
        combined_reconstruction = torch.sum(predicted_values * predicted_masks, dim=1)  # [batch_size, height, width, num_colors]
        
        return {
            'combined_reconstruction': combined_reconstruction,
            'slot_reconstructions': predicted_values,
            'slot_masks': predicted_masks,
            'slots': slots
        }

    def get_individual_segments(self, grid, threshold=0.1):
        """Extract individual segments as separate grids"""
        with torch.no_grad():
            outputs = self.forward(grid)
            slot_masks = outputs['slot_masks']  # [batch_size, num_slots, height, width, 1]
            slot_reconstructions = outputs['slot_reconstructions']  # [batch_size, num_slots, height, width, num_colors]
            
            segments = []
            batch_size = grid.shape[0]
            
            for batch_idx in range(batch_size):
                batch_segments = []
                for slot_idx in range(self.num_slots):
                    mask = slot_masks[batch_idx, slot_idx, :, :, 0]  # [height, width]
                    recon = slot_reconstructions[batch_idx, slot_idx]  # [height, width, num_colors]
                    
                    # Only keep segments with significant attention
                    if mask.max() > threshold:
                        # Get the most likely color for each position
                        segment_grid = torch.argmax(recon, dim=-1)  # [height, width]
                        
                        # Apply mask - set low-attention areas to background (0)
                        segment_grid = torch.where(mask > threshold, segment_grid, torch.zeros_like(segment_grid))
                        
                        batch_segments.append({
                            'grid': segment_grid,
                            'mask': mask,
                            'confidence': mask.max().item()
                        })
                
                segments.append(batch_segments)
            
            return segments
