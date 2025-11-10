"""
Standard softmax attention mechanism.

This module implements the standard quadratic O(N²) softmax attention
used in the baseline Vision Transformer.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from .base import BaseAttention


class SoftmaxAttention(BaseAttention):
    """
    Multi-Head Attention with standard softmax (quadratic O(N²) complexity).

    This is the standard attention mechanism used in vanilla Transformers.
    It computes attention scores as softmax(QK^T/√d)V.

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dropout: Dropout rate
        qkv_bias: Whether to use bias in QKV projection
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False
    ):
        super().__init__(dim, heads, dropout)

        # QKV projection (single matrix for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rpe: Optional[nn.Module] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute softmax attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
            rpe: Optional RPE module (KERPLE not supported for softmax)
            return_attention: If True, also return attention weights

        Returns:
            Output tensor of shape (batch, seq_len, dim)
            Optionally returns attention weights of shape (batch, heads, seq_len, seq_len)

        Raises:
            NotImplementedError: If KERPLE RPE is provided (incompatible with softmax)
        """
        # Check if KERPLE RPE is being used
        if rpe is not None:
            from ..rpe import KERPLEPositionalEncoding
            if isinstance(rpe, KERPLEPositionalEncoding):
                raise NotImplementedError(
                    "KERPLE RPE is designed specifically for kernelized attention "
                    "(FAVOR+/ReLU Performer) and cannot be used with standard softmax attention. "
                    "KERPLE requires linear attention mechanisms to achieve O(n log n) complexity. "
                    "For softmax attention, consider using other RPE types like Shaw et al. 2018."
                )
            # For future RPE types compatible with softmax, we could handle them here
            # For now, reject any RPE
            raise NotImplementedError(
                "RPE integration with softmax attention is not yet implemented. "
                "Currently only KERPLE is implemented, which requires kernelized attention."
            )

        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, heads, N, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Apply mask if provided
        if mask is not None:
            # Mask should be (B, N, N) or (B, 1, N, N)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = (attn @ v)  # (B, heads, N, head_dim)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attention:
            return out, attn
        return out

    def extra_repr(self) -> str:
        """String representation with complexity notation."""
        return super().extra_repr() + ', complexity=O(N²)'