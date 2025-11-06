"""
ReLU-based linear attention mechanism.

This module implements a stub for ReLU attention, which is another
linear complexity attention variant mentioned in the Performer paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from .base import BaseAttention


class ReLUAttention(BaseAttention):
    """
    ReLU-based linear attention (stub implementation).

    This is a placeholder for the ReLU attention variant of the Performer,
    which uses ReLU activation as the kernel function instead of exponential.

    The actual implementation will be added when the specific requirements
    and mathematical formulation are clarified.

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional parameters for future implementation
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(dim, heads, dropout)

        # Store any additional parameters for future use
        self.additional_params = kwargs

        # Placeholder for QKV and projection layers
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Compute ReLU attention (not implemented).

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "ReLU attention is not yet implemented. "
            "This is a placeholder for future development. "
            "Please use 'softmax' or 'favor_plus' attention types instead."
        )

    def extra_repr(self) -> str:
        """String representation."""
        return super().extra_repr() + ', complexity=O(N), status=NOT_IMPLEMENTED'