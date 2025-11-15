"""
Rotary Position Embedding (RoPE) implementation.

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
by Su et al., 2024.

RoPE encodes relative position information by applying rotations to query
and key vectors based on their positions in the sequence.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .base import BaseRPE


class RoPE(BaseRPE):
    """
    Rotary Position Embedding (RoPE).

    RoPE applies rotation-based transformations to Q and K embeddings
    to encode relative positions. The key insight is that rotating Q and K
    by position-dependent angles makes their dot product depend on relative
    position: Q_i^T K_j = f(i-j) for some function f.

    Mathematical Formulation:
        For dimension m, rotation frequency: θ_m = 10000^(-2m/d)
        For position pos, rotation angle: θ_m * pos
        Apply 2D rotation to pairs of dimensions: [x_{2m}, x_{2m+1}]

    Args:
        num_patches: Maximum sequence length (including CLS token)
        dim: Model dimension
        heads: Number of attention heads
        theta: Base for computing rotation frequencies (default: 10000)
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int,
        theta: float = 10000.0,
        **kwargs
    ):
        super().__init__(num_patches, dim, heads)

        self.theta = theta
        self.additional_params = kwargs

        # Compute rotation frequencies for each dimension pair
        # Shape: [head_dim // 2]
        # Formula: θ_m = 10000^(-2m/d) for m = 0, 1, ..., d/2-1
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('freqs', freqs, persistent=False)

        # Pre-compute cos and sin for all positions
        # This is more memory efficient than storing full rotation matrices
        positions = torch.arange(num_patches).float()
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [num_patches, head_dim//2]
        
        cos = torch.cos(angles)  # [num_patches, head_dim//2]
        sin = torch.sin(angles)  # [num_patches, head_dim//2]
        
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor of shape (B, heads, seq_len, head_dim)
            k: Key tensor of shape (B, heads, seq_len, head_dim)
            positions: Optional position indices of shape (seq_len,)
                      If None, uses positions [0, 1, ..., seq_len-1]

        Returns:
            Rotated query and key tensors with same shapes as input
        """
        B, H, N, D = q.shape
        
        # Validate dimensions
        assert D == self.head_dim, f"Expected head_dim={self.head_dim}, got {D}"
        assert H == self.heads, f"Expected heads={self.heads}, got {H}"
        assert N <= self.num_patches, f"Sequence length {N} exceeds max {self.num_patches}"
        
        # Get positions (0-indexed)
        if positions is None:
            pos_indices = torch.arange(N, device=q.device)
        else:
            pos_indices = positions.to(q.device)
            assert pos_indices.max() < self.num_patches, \
                f"Position {pos_indices.max()} exceeds max {self.num_patches}"
        
        # Get cos and sin for these positions
        # cos_cached: [num_patches, head_dim//2]
        # cos_pos: [N, head_dim//2]
        cos_pos = self.cos_cached[pos_indices]  # [N, head_dim//2]
        sin_pos = self.sin_cached[pos_indices]  # [N, head_dim//2]
        
        # Reshape q and k to separate even and odd dimensions
        # q: [B, H, N, D] -> split into [B, H, N, D//2] for even and odd dims
        q_even = q[..., 0::2]  # [B, H, N, D//2] - even indices (0, 2, 4, ...)
        q_odd = q[..., 1::2]   # [B, H, N, D//2] - odd indices (1, 3, 5, ...)
        
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]
        
        # Expand cos and sin to match batch dimensions
        # cos_pos: [N, D//2] -> [1, 1, N, D//2]
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D//2]
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D//2]
        
        # Apply rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        q_rotated_even = q_even * cos_pos - q_odd * sin_pos
        q_rotated_odd = q_even * sin_pos + q_odd * cos_pos
        
        k_rotated_even = k_even * cos_pos - k_odd * sin_pos
        k_rotated_odd = k_even * sin_pos + k_odd * cos_pos
        
        # Interleave back: [even0, odd0, even1, odd1, ...]
        # Stack: [B, H, N, D//2, 2] -> reshape to [B, H, N, D]
        q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1)  # [B, H, N, D//2, 2]
        q_rotated = q_rotated.reshape(B, H, N, D)
        
        k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1)
        k_rotated = k_rotated.reshape(B, H, N, D)
        
        return q_rotated, k_rotated

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward method for BaseRPE interface compatibility.

        Note: RoPE modifies Q and K directly, not attention scores.
        This method is not used in practice. Use apply_rotary_emb() instead.

        Args:
            x: Input tensor (not used in standard RoPE flow)
            attention_scores: Not used for RoPE

        Returns:
            Unchanged input tensor

        Raises:
            NotImplementedError: This method should not be called directly
        """
        # RoPE is applied via apply_rotary_emb() in the attention module
        # This forward() exists only to satisfy the BaseRPE interface
        return x

    def extra_repr(self) -> str:
        """String representation."""
        return super().extra_repr() + f', theta={self.theta}'