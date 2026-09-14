import torch
import torch.nn as nn
import numpy as np
from config import DEVICE, STATIC_DIM, WRIST_DIM

class SpatialConvProjector(nn.Module):
    """
    Downsamples visual sequence representations via a spatial 2D convolutional kernel.
    """
    def __init__(self, embed_dim, grid_size=14):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=grid_size, stride=grid_size)
        
    def forward(self, x):
        # Transpose from sequence format [B, N, C] -> spatial format [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return self.conv(x).flatten(1)

# Initialize production projector components
torch.manual_seed(42)
CONV_PROJECTOR_STATIC = SpatialConvProjector(STATIC_DIM, grid_size=14).to(device=DEVICE, dtype=torch.float16).eval()
CONV_PROJECTOR_WRIST = SpatialConvProjector(WRIST_DIM, grid_size=14).to(device=DEVICE, dtype=torch.float16).eval()

def apply_pooling_variant(sequence_tensor, strategy="conv", has_cls_token=True, is_wrist=False):
    """
    Reshapes sequence tensors into a spatial grid and applies target pooling transformations.
    """
    tokens = sequence_tensor[:, 1:, :] if has_cls_token else sequence_tensor
    B, N, C = tokens.shape
    grid_size = int(np.round(np.sqrt(N)))
    grid_features = tokens.view(B, grid_size, grid_size, C)

    if strategy == "conv":
        projector = CONV_PROJECTOR_WRIST if is_wrist else CONV_PROJECTOR_STATIC
        target_device = next(projector.parameters()).device
        grid_features_dev = grid_features.to(device=target_device, dtype=torch.float16)
        with torch.no_grad():
            return projector(grid_features_dev).squeeze(0).contiguous().float()
            
    elif strategy == "mean":
        return grid_features.mean(dim=(1, 2)).squeeze(0).contiguous().float()
        
    elif strategy == "max":
        val, _ = grid_features.max(dim=2)
        val, _ = val.max(dim=1)
        return val.squeeze(0).contiguous().float()
        
    else:
        raise ValueError(f"Unknown pooling strategy variant: {strategy}")

def compute_backbone_flops_per_frame():
    """
    Analytical FLOP calculation for SigLIP-Base, DINO-ViT, and geometric downsamplers.
    """
    siglip_flops = 12 * (2 * 196 * (768**2) + 4 * 196 * 768 * 3072 + 4 * (196**2) * 768)
    dino_flops = 12 * (2 * 197 * (768**2) + 4 * 197 * 768 * 3072 + 4 * (197**2) * 768)
    projector_flops = 2 * (768 * 14 * 14) * 768 * 1 * 1 * 2
    return siglip_flops + dino_flops + projector_flops