"""
FAVOR+ attention mechanism from the Performer paper.

This module implements linear O(N) attention using positive orthogonal
random features as described in Choromanski et al., 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple
from .base import BaseAttention


class FAVORPlusAttention(BaseAttention):
    """
    FAVOR+ (Fast Attention Via positive Orthogonal Random features).

    This implements the Performer's linear attention mechanism that approximates
    softmax attention using random features, achieving O(N) complexity instead
    of O(N²).

    The key idea is to approximate the softmax kernel using random features:
    softmax(q @ k^T) ≈ φ(q) @ φ(k)^T where φ is a feature map.

    Args:
        dim: Model dimension
        heads: Number of attention heads
        dropout: Dropout rate
        num_features: Number of random features (None for auto d*log(d))
        use_orthogonal: Whether to use orthogonal random features
        feature_redraw_interval: How often to redraw features during training
        qkv_bias: Whether to use bias in QKV projection
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        num_features: Optional[int] = None,
        use_orthogonal: bool = True,
        feature_redraw_interval: Optional[int] = None,
        qkv_bias: bool = False
    ):
        super().__init__(dim, heads, dropout)

        # FAVOR+ specific parameters
        if num_features is None:
            # Auto-compute based on Performer paper recommendation
            num_features = int(self.head_dim * math.log(self.head_dim))
        self.num_features = num_features
        self.use_orthogonal = use_orthogonal
        self.feature_redraw_interval = feature_redraw_interval

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        # Random features for FAVOR+
        self._create_random_features()

        # Redraw counter for training
        self.register_buffer('redraw_counter', torch.tensor(0))

        # Scaling for FAVOR+ (different from standard attention)
        self.favor_scale = self.head_dim ** -0.25  # d^(-1/4) for both Q and K

    def _create_random_features(self):
        """Create random feature matrix Ω for FAVOR+ approximation."""
        # Create random features for each head
        # Shape: (heads, head_dim, num_features)
        if self.use_orthogonal:
            self.register_buffer('omega', self._create_orthogonal_features())
        else:
            omega = torch.randn(self.heads, self.head_dim, self.num_features)
            self.register_buffer('omega', omega)

    def _create_orthogonal_features(self) -> torch.Tensor:
        """
        Create orthogonal random features using QR decomposition.

        This provides lower variance compared to i.i.d. Gaussian features.
        """
        omega_list = []

        for _ in range(self.heads):
            if self.num_features <= self.head_dim:
                # Simple case: can get orthogonal features directly
                gaussian = torch.randn(self.head_dim, self.num_features)
                q, _ = torch.linalg.qr(gaussian, mode='reduced')
                omega = q * math.sqrt(self.head_dim)
            else:
                # Need multiple orthogonal blocks
                num_blocks = math.ceil(self.num_features / self.head_dim)
                blocks = []
                for _ in range(num_blocks):
                    gaussian = torch.randn(self.head_dim, self.head_dim)
                    q, _ = torch.linalg.qr(gaussian, mode='reduced')
                    blocks.append(q)
                omega = torch.cat(blocks, dim=1)[:, :self.num_features]
                omega = omega * math.sqrt(self.head_dim)

            omega_list.append(omega)

        return torch.stack(omega_list, dim=0)  # (heads, head_dim, num_features)

    def _compute_phi_positive(self, x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute positive random features φ⁺(x).

        This is the core of FAVOR+ that ensures positive features for
        approximating the softmax kernel.

        Args:
            x: Input of shape (B, heads, N, head_dim)
            omega: Random features of shape (heads, head_dim, num_features)

        Returns:
            Positive features of shape (B, heads, N, num_features)
        """
        # Project x onto random features: x @ omega
        # (B, heads, N, head_dim) @ (heads, head_dim, num_features) -> (B, heads, N, num_features)
        proj = torch.einsum('bhnd,hdf->bhnf', x, omega)

        # For numerical stability, subtract max before exponential
        proj_max = proj.amax(dim=-1, keepdim=True).detach()
        proj = proj - proj_max

        # Compute norm correction term: ||x||²/2
        x_norm_sq_half = (x ** 2).sum(dim=-1, keepdim=True) / 2.0

        # Positive features: exp(x @ omega - ||x||²/2) / sqrt(num_features)
        phi = torch.exp(proj - x_norm_sq_half) / math.sqrt(self.num_features)

        return phi

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rpe: Optional[nn.Module] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Compute FAVOR+ linear attention with optional RPE.

        Supports two RPE mechanisms:
        - RoPE: Applies rotations to Q/K before feature map, maintains O(N) complexity
        - KERPLE: Uses FFT-based Toeplitz computation, O(N log N) complexity

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask (currently not supported for FAVOR+)
            rpe: Optional RPE module (RoPE or KERPLE)
            return_attention: Not applicable for FAVOR+ (no explicit attention matrix)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        B, N, C = x.shape

        # Optionally redraw features during training
        if self.training and self.feature_redraw_interval is not None:
            if self.redraw_counter % self.feature_redraw_interval == 0:
                self._create_random_features()
            self.redraw_counter += 1

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, heads, N, head_dim)

        # Handle different RPE types
        if rpe is not None:
            from ..rpe import RoPE, KERPLEPositionalEncoding

            if isinstance(rpe, RoPE):
                # RoPE: Apply rotations to Q/K BEFORE feature map
                # RoPE encodes relative position via rotation, works with any attention
                q, k = rpe.apply_rotary_emb(q, k)
                # Then apply standard FAVOR+ scaling
                q = q * self.favor_scale
                k = k * self.favor_scale
            elif isinstance(rpe, KERPLEPositionalEncoding):
                # KERPLE: Requires L2 normalization for training stability
                # From Luo et al., 2021, Section 3.3 and Theorem 3
                q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
                k = k / torch.norm(k, p=2, dim=-1, keepdim=True)
            else:
                # Unknown RPE type - use standard scaling
                q = q * self.favor_scale
                k = k * self.favor_scale
        else:
            # Standard FAVOR+ scaling (d^(-1/4) for both Q and K)
            q = q * self.favor_scale
            k = k * self.favor_scale

        # Compute positive random features
        q_prime = self._compute_phi_positive(q, self.omega)  # (B, heads, N, num_features)
        k_prime = self._compute_phi_positive(k, self.omega)  # (B, heads, N, num_features)

        # Branching based on RPE type
        # KERPLE requires special FFT-based computation
        # RoPE already modified Q/K above, so uses standard path
        from ..rpe import KERPLEPositionalEncoding
        use_kerple = rpe is not None and isinstance(rpe, KERPLEPositionalEncoding)

        if use_kerple:
            # KERPLE attention with FFT-based RPE: O(n log n) complexity
            # From Algorithm 1 in Luo et al., 2021

            # Step 1: Compute D1 = C @ (K'^T @ V) using FFT
            # where C is Toeplitz matrix with C[i,j] = exp(b_{j-i})
            D1 = rpe.apply_rpe_fft(k_prime, v)  # (B, heads, n, num_features, head_dim)

            # Step 2: Compute D2 = C @ K'^T using FFT
            D2 = rpe.apply_rpe_fft(k_prime, None)  # (B, heads, n, num_features)

            # Step 3: Compute attention output
            # For each position i: z_i = (Q'_i @ D1_i) / (Q'_i @ D2_i)

            # Numerator: φ(Q_i) @ D1[i]
            # q_prime: [B, heads, n, num_features]
            # D1: [B, heads, n, num_features, head_dim]
            # We want: for each position i, q_prime[i] @ D1[i]
            out_numerator = torch.einsum('bhnf,bhnfd->bhnd', q_prime, D1)
            # out_numerator: [B, heads, n, head_dim]

            # Denominator: φ(Q_i) @ D2[i]
            # D2: [B, heads, n, num_features]
            out_denominator = torch.einsum('bhnf,bhnf->bhn', q_prime, D2)
            # out_denominator: [B, heads, n]

        else:
            # Standard kernelized attention (no RPE): O(N) complexity
            # Step 1: Compute K'^T @ V
            kv = torch.einsum('bhnf,bhnd->bhfd', k_prime, v)

            # Step 2: Compute Q' @ (K'^T @ V)
            out_numerator = torch.einsum('bhnf,bhfd->bhnd', q_prime, kv)

            # Step 3: Compute normalization denominator
            k_prime_sum = k_prime.sum(dim=2)  # (B, heads, num_features)
            out_denominator = torch.einsum('bhnf,bhf->bhn', q_prime, k_prime_sum)

        # Step 4: Normalize (add small epsilon for stability)
        out = out_numerator / (out_denominator.unsqueeze(-1) + 1e-6)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attention:
            # FAVOR+ doesn't compute explicit attention matrices
            # Could approximate if needed, but would defeat the purpose
            raise NotImplementedError(
                "FAVOR+ doesn't compute explicit attention matrices. "
                "Returning attention weights would require O(N²) computation."
            )

        return out

    def extra_repr(self) -> str:
        """String representation with FAVOR+ parameters."""
        return (
            super().extra_repr() +
            f', complexity=O(N), num_features={self.num_features}, '
            f'orthogonal={self.use_orthogonal}'
        )