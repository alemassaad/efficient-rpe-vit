"""
Unified transformer block that works with any attention mechanism.

This module replaces both TransformerBlock and PerformerTransformerBlock,
eliminating code duplication while supporting different attention types.
"""

import torch
import torch.nn as nn
from typing import Optional


class UnifiedTransformerBlock(nn.Module):
    """
    Universal transformer block with configurable attention.

    This block implements the standard transformer architecture:
    - Pre-norm with LayerNorm
    - Attention with residual connection
    - Feed-forward MLP with residual connection

    The attention mechanism is injected, allowing this single implementation
    to work with softmax attention, FAVOR+, ReLU attention, etc.

    Args:
        dim: Model dimension
        attention: Pre-instantiated attention module
        rpe: Optional pre-instantiated RPE module
        mlp_dim: Hidden dimension of feed-forward network
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        attention: nn.Module,
        rpe: Optional[nn.Module] = None,
        mlp_dim: int = None,
        dropout: float = 0.0
    ):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.mlp_dim = mlp_dim or dim * 4  # Default to 4x expansion

        # Attention components
        self.attention = attention
        self.rpe = rpe

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_dim, dim),
            nn.Dropout(dropout)
        )

        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch, sequence_length, dim)

        Returns:
            Output tensor of same shape as input
        """
        # Pre-norm attention with residual
        normed = self.norm1(x)

        # Apply attention with RPE integrated inside
        # CRITICAL: RPE must be passed INTO attention, not applied after!
        # For KERPLE, this allows O(n log n) FFT-based computation
        if self.rpe is not None:
            attn_output = self.attention(normed, rpe=self.rpe)
        else:
            attn_output = self.attention(normed, rpe=None)

        x = x + attn_output

        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        """String representation of the transformer block."""
        return f'dim={self.dim}, mlp_dim={self.mlp_dim}, has_rpe={self.rpe is not None}'