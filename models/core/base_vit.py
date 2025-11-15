"""
Base Vision Transformer implementation with configurable attention and RPE.

This module provides the core ViT functionality shared by all model variants.
Specific attention mechanisms and RPE implementations are injected via
dependency injection, eliminating code duplication.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Type, Callable
from abc import ABC, abstractmethod


class BaseViT(nn.Module):
    """
    Base Vision Transformer with configurable components.

    This class implements all shared ViT functionality including:
    - Patch embedding
    - Positional encoding
    - CLS token
    - Forward pass logic
    - Weight initialization

    Subclasses or factory functions provide:
    - Attention mechanism
    - Optional RPE mechanism
    - Model-specific parameters

    Args:
        image_size: Input image size (assumes square images)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        patch_size: Size of each patch
        num_classes: Number of output classes
        dim: Model dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        mlp_dim: Hidden dimension of MLP
        dropout: Dropout rate
        attention_builder: Function that creates attention modules
        rpe_builder: Optional function that creates RPE modules
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
        dropout: float = 0.1,
        attention_builder: Optional[Callable] = None,
        rpe_builder: Optional[Callable] = None,
    ):
        super().__init__()

        # Validate inputs
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        assert dim % heads == 0, \
            f"Model dimension {dim} must be divisible by number of heads {heads}"

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Calculate derived dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Build transformer blocks with injected attention
        if attention_builder is None:
            raise ValueError("attention_builder must be provided")

        self.transformer_blocks = nn.ModuleList([
            self._create_transformer_block(
                attention_builder=attention_builder,
                rpe_builder=rpe_builder
            )
            for _ in range(depth)
        ])

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _create_transformer_block(
        self,
        attention_builder: Callable,
        rpe_builder: Optional[Callable] = None
    ) -> nn.Module:
        """
        Create a transformer block with specified attention and optional RPE.

        This method is called for each layer in the transformer.
        """
        # Import here to avoid circular dependency
        from models.components.unified_transformer import UnifiedTransformerBlock

        # Build attention module
        attention = attention_builder(
            dim=self.dim,
            heads=self.heads,
            dropout=self.dropout
        )

        # Build RPE module if provided
        rpe = None
        if rpe_builder is not None:
            # NOTE: num_patches + 1 to account for CLS token
            # After adding CLS token, sequence length is num_patches + 1
            # image_size and patch_size are passed through rpe_config in factory
            rpe = rpe_builder(
                num_patches=self.num_patches + 1,
                dim=self.dim,
                heads=self.heads
            )

        return UnifiedTransformerBlock(
            dim=self.dim,
            attention=attention,
            rpe=rpe,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout
        )

    def _init_weights(self):
        """Initialize model weights using standard initialization schemes."""
        # Initialize positional embedding and CLS token
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers and layer norms throughout the model
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
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Patches tensor of shape (batch, num_patches, patch_dim)
        """
        B, C, H, W = x.shape

        # Validate input dimensions
        assert C == self.in_channels, \
            f"Expected {self.in_channels} channels, got {C}"
        assert H == self.image_size and W == self.image_size, \
            f"Expected {self.image_size}x{self.image_size} images, got {H}x{W}"

        p = self.patch_size

        # Reshape image into patches
        # (B, C, H, W) -> (B, C, H/p, p, W/p, p)
        x = x.reshape(B, C, H // p, p, W // p, p)

        # (B, C, H/p, p, W/p, p) -> (B, H/p, W/p, C, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)

        # (B, H/p, W/p, C, p, p) -> (B, num_patches, patch_dim)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        B = x.shape[0]

        # Convert images to patches
        x = self.patchify(x)  # (B, num_patches, patch_dim)

        # Embed patches
        x = self.patch_embedding(x)  # (B, num_patches, dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, dim)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Extract CLS token for classification
        cls_output = x[:, 0]  # (B, dim)

        # Apply classification head
        return self.mlp_head(cls_output)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts:
            - total: Total number of parameters
            - trainable: Number of trainable parameters
            - non_trainable: Number of non-trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }

    def get_attention_maps(self) -> Any:
        """
        Get attention maps for visualization.

        This is a placeholder that can be overridden by specific implementations.
        """
        raise NotImplementedError(
            "Attention map extraction must be implemented by specific model variants"
        )

    def extra_repr(self) -> str:
        """String representation with key model parameters."""
        return (
            f'image_size={self.image_size}, '
            f'patch_size={self.patch_size}, '
            f'num_patches={self.num_patches}, '
            f'dim={self.dim}, '
            f'depth={self.depth}, '
            f'heads={self.heads}'
        )