"""
Base class for Relative Positional Encoding (RPE) mechanisms.

This module defines the interface for RPE implementations that can be
composed with different attention mechanisms.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseRPE(ABC, nn.Module):
    """
    Abstract base class for Relative Positional Encoding.

    All RPE implementations must inherit from this class and implement
    the forward method. The exact interface will be refined based on
    research of specific RPE papers.

    Args:
        num_patches: Number of patches in the sequence
        dim: Model dimension
        heads: Number of attention heads
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int
    ):
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply relative positional encoding.

        The exact signature will depend on the RPE type:
        - Some RPEs modify attention scores (add bias)
        - Some RPEs modify Q/K embeddings directly
        - Some RPEs work differently

        Args:
            x: Input tensor or attention output
            attention_scores: Optional attention scores for RPEs that modify them

        Returns:
            Modified tensor with RPE applied
        """
        pass

    def get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position matrix.

        Helper method to compute relative positions between all pairs
        of positions in the sequence.

        Args:
            seq_len: Sequence length

        Returns:
            Relative position matrix of shape (seq_len, seq_len)
        """
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        return relative_positions

    def extra_repr(self) -> str:
        """String representation of RPE parameters."""
        return f'num_patches={self.num_patches}, dim={self.dim}, heads={self.heads}'