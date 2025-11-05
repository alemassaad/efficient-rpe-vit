"""
Transformer block module for Vision Transformer.

This module implements a standard transformer block with
multi-head attention and feed-forward network (MLP).
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP.

    Uses pre-norm architecture with residual connections.

    Args:
        dim: Embedding dimension
        heads: Number of attention heads
        mlp_dim: Hidden dimension of the MLP
        dropout: Dropout rate
    """

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(dim, heads, dropout)

        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch, sequence_length, dim)

        Returns:
            Output tensor of shape (batch, sequence_length, dim)
        """
        # Attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x