"""
Baseline Vision Transformer (ViT) model.

This module implements a standard ViT with quadratic O(N²) attention
that serves as the baseline for comparison with efficient Performer variants.
Supports both MNIST (grayscale) and CIFAR-10 (RGB) datasets.
"""

import torch
import torch.nn as nn
from typing import Optional
from .components.transformer import TransformerBlock


class BaselineViT(nn.Module):
    """
    Vision Transformer with standard quadratic attention.

    This is the baseline model with O(N²) complexity that Performer
    variants aim to match in accuracy while being more efficient.

    Args:
        image_size: Size of input image (assumes square)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        patch_size: Size of image patches
        num_classes: Number of output classes
        dim: Embedding dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        mlp_dim: Hidden dimension of MLP in transformer blocks
        dropout: Dropout rate
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Validate inputs
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        assert dim % heads == 0, \
            f"Embedding dimension {dim} must be divisible by number of heads {heads}"

        # Calculate number of patches
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Store attributes for forward pass
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches

        # Patch embedding layer
        self.patch_embedding = nn.Linear(patch_dim, dim)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embedding and CLS token
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Patches tensor of shape (batch, num_patches, patch_dim)
        """
        B, C, H, W = x.shape

        # Validate input dimensions
        assert C == self.in_channels, \
            f"Expected {self.in_channels} input channels, got {C}"
        assert H == self.image_size and W == self.image_size, \
            f"Expected image size {self.image_size}x{self.image_size}, got {H}x{W}"

        p = self.patch_size

        # Reshape image to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)  # (B, num_patches, patch_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        B = x.shape[0]

        # Create patches from input image
        x = self.patchify(x)  # (B, num_patches, patch_dim)

        # Linear projection of patches
        x = self.patch_embedding(x)  # (B, num_patches, dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, dim)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Use CLS token for classification
        cls_output = x[:, 0]  # (B, dim)

        # Classification head
        return self.mlp_head(cls_output)

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers (for visualization).

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            List of attention maps from each transformer block
        """
        # This method can be implemented later for visualization purposes
        raise NotImplementedError("Attention map extraction not yet implemented")

    def count_parameters(self) -> dict:
        """
        Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }


def create_baseline_vit(config: dict) -> BaselineViT:
    """
    Factory function to create BaselineViT from configuration dictionary.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        BaselineViT model instance
    """
    return BaselineViT(
        image_size=config['image_size'],
        in_channels=config['in_channels'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dropout=config.get('dropout', 0.1)
    )