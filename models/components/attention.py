"""
Multi-Head Attention module for Vision Transformer.

This module implements the standard quadratic O(N²) softmax attention
that serves as the baseline for comparison with efficient Performer variants.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with quadratic O(N²) complexity.

    This is the standard attention used in the baseline ViT model.
    Uses Q @ K^T computation followed by softmax normalization.

    Args:
        dim: Embedding dimension
        heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # Single matrix for Q, K, V projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, sequence_length, dim)

        Returns:
            Output tensor of shape (batch, sequence_length, dim)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: Q @ K^T / sqrt(d) - O(N²) complexity
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x