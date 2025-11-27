"""
Circulant-STRING Relative Position Encoding.

Implementation based on "Learning the RoPEs: Better 2D and 3D Position Encodings
with STRING" (Schenck et al., 2025) - arXiv:2502.02562

Circulant-STRING applies position-dependent rotations to query and key vectors
using the matrix exponential of circulant skew-symmetric generators. This is
computed efficiently via FFT in O(d log d) time per token.

Mathematical Background
-----------------------
STRING (Separable Translationally Invariant Position Encodings) defines position
encoding as a mapping from position vectors to rotation matrices:

    R(r_i) = exp(Σ_k L_k · [r_i]_k)

Where:
- r_i ∈ R^{d_c} is the position vector (e.g., 2D coordinates for images)
- L_k are learnable, commuting, skew-symmetric generator matrices
- exp(·) is the matrix exponential

For Circulant-STRING, each generator L_k = C_k - C_k^T where C_k is a circulant
matrix parameterized by d scalars. This construction ensures:
1. L_k is skew-symmetric: L_k^T = -L_k
2. L_k matrices commute (circulant matrices always commute)
3. exp(L_k) is orthogonal (rotation matrix)

FFT-Based Computation (Theorem 3.5 from paper)
----------------------------------------------
Circulant matrices are diagonalized by the DFT matrix:
    C = F^{-1} · diag(λ) · F

Where λ = FFT(c) for first row c of C.

For L = C - C^T:
    λ_L = FFT(c) - conj(FFT(c)) = 2i · Im(FFT(c))

The eigenvalues are purely imaginary (as expected for skew-symmetric matrices).

To apply exp(r · L) · z efficiently:
    1. Compute λ_L = FFT(c) - conj(FFT(c))
    2. Scale: μ = r · λ_L
    3. Apply: z' = IFFT(exp(μ) · FFT(z))

Since μ is purely imaginary, exp(μ) = exp(iθ) lies on the unit circle,
so there is no numerical overflow risk.

References
----------
- Schenck et al. (2025). "Learning the RoPEs: Better 2D and 3D Position
  Encodings with STRING". arXiv:2502.02562
- Su et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position
  Embedding". Neurocomputing.
- Choromanski et al. (2020). "Rethinking Attention with Performers".
  arXiv:2009.14794
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .base import BaseRPE


class CirculantStringRPE(BaseRPE):
    """
    Circulant-STRING Relative Position Encoding.

    Applies position-dependent rotation to queries and keys via FFT-based
    matrix exponential of circulant skew-symmetric generators. Compatible
    with both linear attention (Performers) and standard softmax attention.

    Key equation:
        R(r_i) = exp(Σ_k L_k · [r_i]_k)

    Where L_k = C_k - C_k^T (skew-symmetric circulant matrix).

    The rotation is applied to Q and K independently (separable), ensuring
    that attention scores depend only on relative positions:
        q_i^T · k_j = q^T · R(r_i)^T · R(r_j) · k = q^T · R(r_j - r_i) · k

    CLS Token Handling
    ------------------
    Following the official RoPE-ViT implementation (naver-ai/rope-vit), the
    CLS token at index 0 does NOT receive position rotation. Only patch
    tokens (indices 1 to N-1) are rotated. This is because CLS has no
    meaningful spatial position in the image grid.

    Args:
        num_patches: Total sequence length including CLS token
        dim: Model dimension (total, not per-head)
        heads: Number of attention heads
        coord_dim: Coordinate dimensionality (2 for 2D images, 3 for 3D, etc.)
        block_size: Optional block size for block-circulant optimization.
                    If None, uses full head_dim (default). Future optimization.
        **kwargs: Additional parameters for compatibility

    Attributes:
        circulant_coeffs: Learnable parameters of shape (heads, coord_dim, head_dim)
                         or (heads, coord_dim, num_blocks, block_size) if block_size set
        patch_positions: Precomputed 2D grid positions for patches

    Example:
        >>> rpe = CirculantStringRPE(num_patches=197, dim=768, heads=12, coord_dim=2)
        >>> q = torch.randn(2, 12, 197, 64)  # (B, H, N, D)
        >>> k = torch.randn(2, 12, 197, 64)
        >>> q_rot, k_rot = rpe.apply_circulant_string(q, k)
        >>> # CLS token unchanged: q_rot[:, :, 0] == q[:, :, 0]
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int,
        coord_dim: int = 2,
        block_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(num_patches, dim, heads)

        self.coord_dim = coord_dim
        self.block_size = block_size
        self.additional_params = kwargs

        # Validate block_size (future optimization)
        if block_size is not None:
            if self.head_dim % block_size != 0:
                raise ValueError(
                    f"head_dim ({self.head_dim}) must be divisible by "
                    f"block_size ({block_size})"
                )
            self.num_blocks = self.head_dim // block_size
            # TODO: Implement block-circulant optimization
            # For now, fall back to full-dimension circulant
            import warnings
            warnings.warn(
                f"block_size={block_size} specified but block-circulant "
                "optimization not yet implemented. Using full-dimension "
                "circulant. This will be added in a future update.",
                UserWarning
            )
            self.block_size = None  # Fall back to full dimension

        # Learnable circulant coefficients: (heads, coord_dim, head_dim)
        # Each head has separate coefficients for each coordinate dimension
        # These parameterize the first row of circulant matrices C_k
        self.circulant_coeffs = nn.Parameter(
            torch.zeros(heads, coord_dim, self.head_dim)
        )

        # Initialize with small values near zero for identity-like initial transform
        # This follows the paper's recommendation for stable training
        nn.init.normal_(self.circulant_coeffs, mean=0.0, std=0.01)

        # Precompute and cache 2D positions for patches
        self._setup_positions(num_patches)

    def _setup_positions(self, num_patches: int) -> None:
        """
        Precompute 2D grid positions for image patches.

        Positions are integer coordinates (not normalized), following the
        convention from RoPE-ViT: p_n^x ∈ {0, 1, ..., W-1}, p_n^y ∈ {0, 1, ..., H-1}

        The CLS token at index 0 has no position (excluded from rotation).
        Patch tokens start at index 1 and are arranged row by row.

        Args:
            num_patches: Total sequence length including CLS token
        """
        # num_patches includes CLS token, so actual patches = num_patches - 1
        num_patch_tokens = num_patches - 1

        if num_patch_tokens <= 0:
            # Edge case: no patch tokens (only CLS)
            self.register_buffer('patch_positions', torch.zeros(0, self.coord_dim))
            self._patches_per_side = 0
            return

        # Compute patches per side (assuming square image)
        patches_per_side = int(math.sqrt(num_patch_tokens))

        if patches_per_side ** 2 != num_patch_tokens:
            raise ValueError(
                f"num_patches - 1 = {num_patch_tokens} must be a perfect square "
                f"for 2D position encoding. Got sqrt ≈ {math.sqrt(num_patch_tokens):.2f}"
            )

        self._patches_per_side = patches_per_side

        # Create 2D grid coordinates (x, y) for each patch
        # Patches are arranged row by row: (0,0), (1,0), (2,0), ..., (0,1), (1,1), ...
        y_coords = torch.arange(patches_per_side, dtype=torch.float32)
        x_coords = torch.arange(patches_per_side, dtype=torch.float32)

        # meshgrid with 'ij' indexing: yy[i,j] = i, xx[i,j] = j
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # positions: (num_patch_tokens, 2) with [x, y] for each patch
        # Flatten in row-major order to match how patches are flattened from image
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        self.register_buffer('patch_positions', positions)

    def get_eigenvalues(self) -> torch.Tensor:
        """
        Compute eigenvalues of L = C - C^T from circulant coefficients.

        For a circulant matrix C with first row c = [c_0, c_1, ..., c_{d-1}]:
            eigenvalues of C = FFT(c)
            eigenvalues of C^T = conj(FFT(c))  [since C^T is also circulant]
            eigenvalues of L = C - C^T = FFT(c) - conj(FFT(c)) = 2i · Im(FFT(c))

        The eigenvalues are purely imaginary, as expected for a skew-symmetric matrix.

        Returns:
            eigenvalues: Complex tensor of shape (heads, coord_dim, head_dim)
                        with purely imaginary values (real part ≈ 0)
        """
        # FFT of coefficients gives eigenvalues of circulant matrix C
        # circulant_coeffs: (heads, coord_dim, head_dim)
        lambda_c = torch.fft.fft(self.circulant_coeffs, dim=-1)

        # Eigenvalues of C^T are complex conjugates of eigenvalues of C
        # (This follows from: C^T has first row [c_0, c_{d-1}, ..., c_1])
        # Eigenvalues of L = C - C^T
        eigenvalues = lambda_c - torch.conj(lambda_c)

        # Result: eigenvalues = 2i * Im(lambda_c), which is purely imaginary
        return eigenvalues

    def apply_rotation(
        self,
        x: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply position-dependent rotation to input tensor.

        Computes: x' = R(r) · x = exp(Σ_k L_k · r_k) · x

        Using FFT-based computation:
            1. μ = Σ_k r_k · λ_{L_k}  (combine scaled eigenvalues)
            2. x' = IFFT(exp(μ) · FFT(x))

        Since μ is purely imaginary (μ = iθ), exp(μ) = cos(θ) + i·sin(θ)
        lies on the unit circle, so there's no numerical overflow risk.

        Args:
            x: Input tensor of shape (B, H, N, D) - queries or keys
               N is the number of tokens EXCLUDING CLS
            positions: Position coordinates of shape (N, coord_dim)
                      Integer coordinates for each token

        Returns:
            x_rotated: Rotated tensor of same shape (B, H, N, D)
        """
        B, H, N, D = x.shape

        # Get eigenvalues: (H, coord_dim, D) - purely imaginary
        eigenvalues = self.get_eigenvalues()

        # Scale eigenvalues by positions and sum across coordinates
        # This computes: μ = Σ_k position_k · eigenvalue_k

        # positions: (N, coord_dim) -> (1, 1, N, coord_dim, 1)
        pos = positions.to(x.device).view(1, 1, N, self.coord_dim, 1)

        # eigenvalues: (H, coord_dim, D) -> (1, H, 1, coord_dim, D)
        eig = eigenvalues.view(1, H, 1, self.coord_dim, D)

        # Weighted sum across coordinates: (1, H, N, D)
        # μ_combined[h, n, d] = Σ_k pos[n, k] * eig[h, k, d]
        mu_combined = (pos * eig).sum(dim=-2)  # (1, H, N, D)

        # Apply rotation via FFT
        # Since μ is purely imaginary, exp(μ) is on unit circle (no overflow)

        # Convert x to complex for FFT
        x_complex = x.to(torch.complex64)

        # Transform to frequency domain
        x_freq = torch.fft.fft(x_complex, dim=-1)

        # Apply rotation: element-wise multiply by exp(μ)
        # exp(μ) where μ is purely imaginary gives rotation on unit circle
        x_freq_rotated = torch.exp(mu_combined) * x_freq

        # Transform back to spatial domain and take real part
        # Result should be real (imaginary part is numerical noise)
        x_rotated = torch.fft.ifft(x_freq_rotated, dim=-1).real

        return x_rotated.to(x.dtype)

    def apply_circulant_string(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Circulant-STRING rotation to queries and keys.

        Following the official RoPE-ViT implementation, the CLS token at
        index 0 is NOT rotated. Only patch tokens (indices 1 to N-1)
        receive position-dependent rotation.

        Args:
            q: Query tensor of shape (B, H, N, D) including CLS at index 0
            k: Key tensor of shape (B, H, N, D) including CLS at index 0

        Returns:
            q_rotated: Rotated queries, same shape. CLS unchanged.
            k_rotated: Rotated keys, same shape. CLS unchanged.

        Note:
            Values (V) are NOT transformed - only Q and K receive position
            encoding, as per the STRING paper.
        """
        # Handle edge case: sequence has only CLS token (no patches)
        if q.shape[2] <= 1:
            return q, k

        # Separate CLS token from patch tokens
        # CLS is at index 0, patches are at indices 1:N
        q_cls = q[:, :, :1, :]      # (B, H, 1, D)
        q_patches = q[:, :, 1:, :]  # (B, H, N-1, D)

        k_cls = k[:, :, :1, :]      # (B, H, 1, D)
        k_patches = k[:, :, 1:, :]  # (B, H, N-1, D)

        # Apply rotation only to patch tokens
        q_patches_rotated = self.apply_rotation(q_patches, self.patch_positions)
        k_patches_rotated = self.apply_rotation(k_patches, self.patch_positions)

        # Concatenate CLS token back (unchanged)
        q_rotated = torch.cat([q_cls, q_patches_rotated], dim=2)
        k_rotated = torch.cat([k_cls, k_patches_rotated], dim=2)

        return q_rotated, k_rotated

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        BaseRPE interface method - not used directly for Circulant-STRING.

        Circulant-STRING modifies Q and K via apply_circulant_string(),
        not attention scores. This method exists for interface compatibility.

        For actual usage, call apply_circulant_string(q, k) from attention modules.

        Args:
            x: Input tensor (unused)
            attention_scores: Attention scores (unused for this RPE type)

        Returns:
            Unchanged input tensor
        """
        # Circulant-STRING operates on Q/K, not on x or attention scores
        # This forward() exists only for BaseRPE interface compatibility
        return x

    def extra_repr(self) -> str:
        """String representation with Circulant-STRING parameters."""
        base = super().extra_repr()
        return (
            f'{base}, coord_dim={self.coord_dim}, '
            f'patches_per_side={self._patches_per_side}, '
            f'params_per_head={self.coord_dim * self.head_dim}'
        )


# =============================================================================
# Future Optimization: Block-Circulant Structure
# =============================================================================
#
# The STRING paper mentions sweeping block sizes {4, 8, 16, 32, 64} for
# efficiency, with optimal often around 16. Block-circulant structure
# divides head_dim into blocks and applies separate circulant matrices
# to each block.
#
# Benefits:
# - Reduced parameters: H × d_c × num_blocks × block_size (vs H × d_c × D)
# - Potentially faster FFT on smaller dimensions
# - May provide regularization benefits
#
# Implementation plan:
# 1. Add block_size parameter (validated in __init__)
# 2. Reshape circulant_coeffs to (H, d_c, num_blocks, block_size)
# 3. Apply FFT per block instead of full dimension
# 4. Benchmark to find optimal block_size per dataset
#
# This optimization is left for future work.
# =============================================================================
