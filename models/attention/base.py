"""
Base class for all attention mechanisms.

This module defines the interface that all attention implementations must follow,
ensuring compatibility with the unified transformer architecture.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseAttention(ABC, nn.Module):
    """
    Abstract base class for attention mechanisms.

    All attention implementations (Softmax, FAVOR+, ReLU, etc.) must inherit
    from this class and implement the forward method with a consistent signature.

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0
    ):
        super().__init__()

        assert dim % heads == 0, \
            f"Model dimension {dim} must be divisible by heads {heads}"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Compute attention output.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights (if applicable)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
            Optionally returns attention weights if supported and requested
        """
        pass

    def extra_repr(self) -> str:
        """String representation of attention parameters."""
        return f'dim={self.dim}, heads={self.heads}, head_dim={self.head_dim}'