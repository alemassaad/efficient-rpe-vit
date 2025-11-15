"""
Circulant-STRING RPE implementation.

Based on circulant-STRING positional encoding from Schenck et al., 2025.

Circulant-STRING uses a circulant matrix structure to parameterize relative
position biases efficiently. The circulant structure allows efficient computation
and reduces the number of parameters needed for relative position encoding.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .base import BaseRPE


class CirculantStringRPE(BaseRPE):
    """
    Circulant-STRING Relative Position Encoding.

    This implementation uses a circulant matrix structure to encode relative
    position biases. For 2D vision transformers, we handle both row and column
    relative positions.

    The circulant structure means that relative position bias B[i,j] depends
    only on the relative position (j-i) mod sequence_length, allowing efficient
    parameterization with O(n) parameters instead of O(n²).

    Args:
        num_patches: Maximum sequence length (including CLS token)
        dim: Model dimension
        heads: Number of attention heads
        image_size: Optional image size for 2D position encoding
        patch_size: Optional patch size for 2D position encoding
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(num_patches, dim, heads)

        self.additional_params = kwargs
        
        # Determine if we're using 2D position encoding
        self.use_2d = (image_size is not None and patch_size is not None)
        
        if self.use_2d:
            # For 2D: compute patches per row/column
            self.patches_per_side = int(math.sqrt(num_patches - 1))  # -1 for CLS token
            assert self.patches_per_side ** 2 == (num_patches - 1), \
                f"num_patches-1={num_patches-1} must be a perfect square for 2D encoding"
            
            # Learnable biases for row and column relative positions
            # Each head has separate biases for row and column offsets
            # Max relative position: patches_per_side - 1 in each direction
            max_rel_pos = 2 * self.patches_per_side - 1
            self.rel_pos_bias_row = nn.Parameter(
                torch.zeros(heads, max_rel_pos)
            )
            self.rel_pos_bias_col = nn.Parameter(
                torch.zeros(heads, max_rel_pos)
            )
            
            # Initialize with small values
            nn.init.normal_(self.rel_pos_bias_row, mean=0.0, std=0.02)
            nn.init.normal_(self.rel_pos_bias_col, mean=0.0, std=0.02)
        else:
            # For 1D: simple circulant structure
            # Learnable biases for relative positions
            # Max relative position: num_patches - 1
            max_rel_pos = 2 * num_patches - 1
            self.rel_pos_bias = nn.Parameter(
                torch.zeros(heads, max_rel_pos)
            )
            
            # Initialize with small values
            nn.init.normal_(self.rel_pos_bias, mean=0.0, std=0.02)

    def _get_2d_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Convert 1D sequence indices to 2D (row, col) positions.
        
        Assumes CLS token is at position 0, then patches are arranged
        row by row.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            Position tensor of shape (seq_len, 2) with [row, col] for each position
        """
        positions = torch.zeros(seq_len, 2, dtype=torch.long, device=device)
        
        # Position 0 is CLS token (special handling)
        # Positions 1 to seq_len-1 are patches
        for i in range(1, seq_len):
            patch_idx = i - 1  # 0-indexed patch
            row = patch_idx // self.patches_per_side
            col = patch_idx % self.patches_per_side
            positions[i, 0] = row
            positions[i, 1] = col
        
        return positions

    def apply_bias(
        self,
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply relative position bias to attention scores.

        Args:
            attention_scores: Attention scores of shape (B, heads, seq_len, seq_len)
            seq_len: Optional sequence length (if None, inferred from attention_scores)

        Returns:
            Attention scores with relative position bias added
        """
        B, H, N, M = attention_scores.shape
        assert N == M, "Attention scores must be square"
        
        if seq_len is None:
            seq_len = N
        
        # Create relative position bias matrix
        if self.use_2d:
            bias = self._compute_2d_bias(seq_len)  # [heads, seq_len, seq_len]
        else:
            bias = self._compute_1d_bias(seq_len)  # [heads, seq_len, seq_len]
        
        # Add bias: [B, H, N, N] + [H, N, N] -> [B, H, N, N]
        attention_scores = attention_scores + bias.unsqueeze(0)
        
        return attention_scores

    def _compute_1d_bias(self, seq_len: int) -> torch.Tensor:
        """
        Compute 1D relative position bias matrix using circulant structure.

        Args:
            seq_len: Sequence length

        Returns:
            Bias matrix of shape (heads, seq_len, seq_len)
        """
        # Compute relative positions: rel_pos[i, j] = j - i
        positions = torch.arange(seq_len, device=self.rel_pos_bias.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        
        # Map to bias indices: center at num_patches-1
        # rel_pos ranges from -(seq_len-1) to (seq_len-1)
        # bias index = rel_pos + (num_patches - 1)
        max_rel_pos = self.rel_pos_bias.shape[1]
        center = max_rel_pos // 2
        
        # Clamp to valid range
        bias_indices = rel_pos + center
        bias_indices = torch.clamp(bias_indices, 0, max_rel_pos - 1)
        
        # Lookup biases: [heads, seq_len, seq_len]
        bias = self.rel_pos_bias[:, bias_indices]  # [heads, seq_len, seq_len]
        
        return bias

    def _compute_2d_bias(self, seq_len: int) -> torch.Tensor:
        """
        Compute 2D relative position bias matrix.

        For 2D patches, we combine row and column relative position biases.

        Args:
            seq_len: Sequence length

        Returns:
            Bias matrix of shape (heads, seq_len, seq_len)
        """
        # Get device from parameters
        device = self.rel_pos_bias_row.device
        
        # Get 2D positions
        pos_2d = self._get_2d_positions(seq_len, device)  # [seq_len, 2]
        
        # Compute relative positions
        # rel_row[i, j] = row[j] - row[i]
        # rel_col[i, j] = col[j] - col[i]
        rel_row = pos_2d[:, 0:1] - pos_2d[:, 0:1].t()  # [seq_len, seq_len]
        rel_col = pos_2d[:, 1:2] - pos_2d[:, 1:2].t()  # [seq_len, seq_len]
        
        # Map to bias indices
        max_rel_pos = self.rel_pos_bias_row.shape[1]
        center = max_rel_pos // 2
        
        # Clamp to valid range
        bias_idx_row = torch.clamp(rel_row + center, 0, max_rel_pos - 1)
        bias_idx_col = torch.clamp(rel_col + center, 0, max_rel_pos - 1)
        
        # Lookup biases for each head
        # [heads, seq_len, seq_len]
        bias_row = self.rel_pos_bias_row[:, bias_idx_row]
        bias_col = self.rel_pos_bias_col[:, bias_idx_col]
        
        # Combine row and column biases (additive)
        bias = bias_row + bias_col
        
        return bias

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply circulant-STRING RPE to attention scores.

        Args:
            x: Input tensor (not used, kept for interface compatibility)
            attention_scores: Attention scores of shape (B, heads, seq_len, seq_len)

        Returns:
            Attention scores with relative position bias added

        Raises:
            ValueError: If attention_scores is None
        """
        if attention_scores is None:
            raise ValueError(
                "Circulant-STRING RPE requires attention_scores. "
                "This RPE modifies attention scores, not embeddings."
            )
        
        return self.apply_bias(attention_scores)

    def extra_repr(self) -> str:
        """String representation."""
        mode = "2D" if self.use_2d else "1D"
        return super().extra_repr() + f', mode={mode}'