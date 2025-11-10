"""
KERPLE: Kernelized Attention with Relative Positional Encoding.

Based on "Stable, Fast and Accurate: Kernelized Attention with Relative
Positional Encoding" by Luo et al., NeurIPS 2021.

This RPE mechanism is specifically designed for kernelized attention
(FAVOR+/ReLU Performer) and achieves O(n log n) complexity using FFT.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseRPE
from .fft_utils import fft_toeplitz_matmul


class KERPLEPositionalEncoding(BaseRPE):
    """
    KERPLE: Kernelized attention with RPE via FFT.

    This implements the Most General RPE from Luo et al., 2021 (Algorithm 1).
    It uses learnable scalar biases for relative positions and applies them
    efficiently using FFT-based Toeplitz matrix multiplication.

    Key Properties:
        - Complexity: O(n log n) via FFT (vs O(n²) for naive RPE)
        - Designed for kernelized attention (FAVOR+/ReLU)
        - Requires Q/K normalization for training stability
        - Simple parameters: scalar biases b_{j-i} for each relative position

    Mathematical Formulation:
        z_i = φ(Q_i) Σ_j exp(b_{j-i}) φ(K_j)^T V_j / φ(Q_i) Σ_j exp(b_{j-i}) φ(K_j)^T

    Where:
        - b_{j-i}: Learnable scalar bias for relative position (j-i)
        - φ(·): Kernelized feature map (PRF or TRF)
        - Toeplitz matrix C[i,j] = exp(b_{j-i}) enables FFT acceleration

    Args:
        num_patches: Sequence length n (number of patches/tokens)
        dim: Model dimension
        heads: Number of attention heads

    Reference:
        Luo et al., "Stable, Fast and Accurate: Kernelized Attention with
        Relative Positional Encoding", NeurIPS 2021, Algorithm 1 (page 7)
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int
    ):
        super().__init__(num_patches, dim, heads)

        # Learnable relative position biases
        # For sequence length n, we need biases for relative positions:
        # -(n-1), -(n-2), ..., -1, 0, 1, ..., (n-2), (n-1)
        # Total: 2n - 1 biases
        max_rel_pos = 2 * num_patches - 1

        # Each attention head gets its own set of biases
        # Shape: [heads, 2n-1]
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(heads, max_rel_pos)
        )

        # Initialize with small values for stability
        # Following standard practice for positional encodings
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=0.02)

        # Store for reference
        self.max_rel_pos = max_rel_pos

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        This forward method is not used for KERPLE.

        KERPLE must be integrated inside kernelized attention computation
        via apply_rpe_fft(). This method exists only to satisfy the BaseRPE
        interface.

        Raises:
            NotImplementedError: Always, as KERPLE uses apply_rpe_fft() instead
        """
        raise NotImplementedError(
            "KERPLE does not use the standard forward() interface. "
            "Use apply_rpe_fft() method instead, which must be called "
            "from within kernelized attention computation (FAVOR+/ReLU). "
            "See models/attention/favor_plus.py for usage example."
        )

    def apply_rpe_fft(
        self,
        k_prime: torch.Tensor,
        v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply RPE via FFT-accelerated Toeplitz multiplication.

        This method computes the two key quantities needed for KERPLE attention:
        - D1 = C @ (K'^T @ V)  [when v is provided]
        - D2 = C @ K'^T         [when v is None]

        Where C is the Toeplitz matrix with C[i,j] = exp(b_{j-i}).

        The FFT algorithm reduces complexity from O(n²) to O(n log n).

        Args:
            k_prime: Kernelized keys φ(K), shape [B, heads, n, num_features]
                     Output from kernelized feature map (PRF or TRF)
            v: Values V, shape [B, heads, n, head_dim]
               If provided, computes D1 = C @ (K'^T @ V)
               If None, computes D2 = C @ K'^T for normalization

        Returns:
            If v provided: D1 of shape [B, heads, n, num_features, head_dim]
            If v is None: D2 of shape [B, heads, n, num_features]

        Mathematical Details:
            From Equations 11-13 in the paper:

            D1[i] = Σ_j C[i,j] * (K'[j]^T @ V[j])
                  = (C @ (K'^T @ V))[i]

            D2[i] = Σ_j C[i,j] * K'[j]^T
                  = (C @ K'^T)[i]

            Where C is the Toeplitz matrix enables FFT multiplication.

        Example:
            >>> # In kernelized attention:
            >>> q_prime = phi(Q)  # [B, heads, n, m]
            >>> k_prime = phi(K)  # [B, heads, n, m]
            >>> v = V             # [B, heads, n, d]
            >>>
            >>> # Compute with RPE
            >>> D1 = rpe.apply_rpe_fft(k_prime, v)      # [B, heads, m, d]
            >>> D2 = rpe.apply_rpe_fft(k_prime, None)   # [B, heads, m, n]
            >>>
            >>> # Final attention
            >>> numerator = q_prime @ D1       # [B, heads, n, d]
            >>> denominator = (q_prime @ D2).sum(-1)  # [B, heads, n]
            >>> output = numerator / (denominator.unsqueeze(-1) + eps)
        """
        B, H, N, F = k_prime.shape

        # Verify number of heads matches
        assert H == self.heads, f"Expected {self.heads} heads, got {H}"

        # Compute Toeplitz coefficients: c_k = exp(b_k)
        # Shape: [heads, 2n-1]
        c = torch.exp(self.rel_pos_bias)

        if v is not None:
            # Compute D1 = C @ (K'^T @ V)
            # This is used for the attention numerator

            # Step 1: Compute K'^T @ V
            # k_prime: [B, heads, n, num_features]
            # v: [B, heads, n, head_dim]
            # We want: [B, heads, num_features, n] @ [B, heads, n, head_dim]
            #        = [B, heads, num_features, head_dim]
            k_prime_t = k_prime.transpose(2, 3)  # [B, heads, num_features, n]
            kv = torch.einsum('bhfn,bhnd->bhfd', k_prime_t, v)
            # kv shape: [B, heads, num_features, head_dim]

            # Step 2: Apply Toeplitz multiplication: D1 = C @ kv
            # For each head, for each feature dimension:
            # C: [n, n] Toeplitz matrix
            # kv[feature_idx]: [n, head_dim] needs to become [n] for each d

            # We need to apply C (n×n) to kv treated as matrix [n, num_features*head_dim]
            # But our FFT function expects [n, d] where we apply to each column

            # Reshape kv: [B, heads, num_features, head_dim] -> [B*heads, n, num_features*head_dim]
            # Wait, that's wrong. Let me reconsider...

            # The Toeplitz matrix acts on the n-dimension.
            # kv is [B, heads, num_features, head_dim]
            # We need C @ kv where C is [n, n] but kv has n in a different position

            # Actually, looking at the paper more carefully:
            # D1[i] = vec(Σ_j e^{b_{j-i}} φ(K_j)^T V_j)
            # This is applying the Toeplitz to each row of K'^T @ V

            # Let me re-examine: K'^T @ V is [num_features, n] @ [n, head_dim] = [num_features, head_dim]
            # Wait no, dimensions: k_prime is [B, heads, n, num_features]
            # So K'^T is [B, heads, num_features, n]

            # Then K'^T @ V: [num_features, n] @ [n, head_dim] = [num_features, head_dim] per batch/head
            # And we need to apply Toeplitz to this in the "n" dimension

            # I think the issue is: we want C @ (K'^T @ V) where the multiplication
            # is along the first dimension of (K'^T @ V)

            # Let's think differently: vectorize the (K'^T @ V) result
            # K'^T @ V for a single batch/head: [num_features, head_dim]
            # We want to apply C[i,j] to this matrix...

            # From the paper equation 11:
            # D_tilde_1 has rows: vec(Σ_j e^{b_{j-1}} φ(K_j)^T V_j), ..., vec(Σ_j e^{b_{j-n}} φ(K_j)^T V_j)

            # OK so for each position i in [1...n], we compute:
            # Σ_j e^{b_{j-i}} φ(K_j)^T V_j
            # This is a weighted sum over j (sequence positions) of the outer product φ(K_j)^T V_j

            # φ(K_j)^T is [num_features, 1] and V_j is [1, head_dim]
            # So φ(K_j)^T V_j is [num_features, head_dim]

            # For position i, we sum these with weights e^{b_{j-i}}
            # That's exactly a Toeplitz matrix multiplication!

            # So we need to apply Toeplitz in the "j" dimension to get result in "i" dimension

            # Current kv: [B, heads, num_features, head_dim]
            # This represents K'^T @ V where the n dimension was contracted

            # But we need to UN-contract it and apply Toeplitz
            # Actually wait - K'^T @ V is not what we want!

            # Let me re-read the paper... Equation 11:
            # D_tilde_1 has i-th row as vec(Σ_j e^{b_{j-i}} φ(x_j W^K)^T (x_j W^V))

            # So for each i, we have a sum over j. Let's denote:
            # A1[j] = φ(K_j)^T V_j which is [num_features x head_dim] matrix
            # Then D1[i] = Σ_j C[i,j] * A1[j]

            # This is matrix-valued Toeplitz multiplication!
            # We have n matrices A1[1], ..., A1[n] each of size [num_features, head_dim]
            # And we want to compute weighted sums with Toeplitz weights

            # This can be done by vectorizing: treat each of the num_features*head_dim elements separately
            # Reshape A1: [n, num_features, head_dim] -> [n, num_features*head_dim]
            # Apply Toeplitz: [n, n] @ [n, num_features*head_dim] = [n, num_features*head_dim]
            # Reshape back: [n, num_features*head_dim] -> [n, num_features, head_dim]

            # In our current code:
            # k_prime: [B, heads, n, num_features]
            # v: [B, heads, n, head_dim]
            # We want A1[j] = k_prime[:, :, j, :]^T @ v[:, :, j, :]
            # = [num_features, 1] @ [1, head_dim] = [num_features, head_dim]

            # So A1 stacked: [B, heads, n, num_features, head_dim]
            # via outer product
            A1 = torch.einsum('bhkf,bhkd->bhkfd', k_prime, v)
            # A1: [B, heads, n, num_features, head_dim]

            # Reshape for FFT: [B, heads, n, num_features*head_dim]
            A1_flat = A1.reshape(B, H, N, F * v.shape[-1])

            # Apply Toeplitz for each head (vectorized over batch)
            # Use the vectorized FFT function to process all batches at once
            results = []
            for h in range(H):
                # c[h]: [2n-1]
                # A1_flat[:, h]: [B, n, num_features*head_dim]
                # Vectorized: processes all B batches in parallel
                D1_h = fft_toeplitz_matmul(c[h], A1_flat[:, h])  # [B, n, F*D]
                results.append(D1_h)

            # Stack and reshape: [B, heads, n, num_features*head_dim] -> [B, heads, n, num_features, head_dim]
            D1_flat = torch.stack(results, dim=1)  # [B, heads, n, F*D]
            D1 = D1_flat.reshape(B, H, N, F, v.shape[-1])  # [B, heads, n, F, D]

            # Transpose to match expected output: [B, heads, num_features, head_dim]
            # Wait, the output should be per-position... let me reconsider

            # From the algorithm in the paper (page 7):
            # D1 ← FFTMatrixMul(c, A1)
            # where A1 = [φ(K_i)^T V_i] for i=1...n

            # Then X ← [φ(Q_i) D1_i / φ(Q_i) D2_i] for i=1...n

            # So D1 has n rows, one per position i
            # D1[i] = Σ_j C[i,j] A1[j]

            # And then φ(Q_i) D1_i means we matrix-multiply:
            # φ(Q_i): [1, num_features]
            # D1[i]: [num_features, head_dim]
            # Result: [1, head_dim]

            # So D1 should be [B, heads, n, num_features, head_dim] - WRONG
            # Actually D1[i] is vec(...) in the paper
            # Hmm, let's look at the algorithm again

            # Actually, in Algorithm 1:
            # The output X is computed as: X ← [φ(Q_i)D1_i / φ(Q_i)D2_i] for i=1...n
            # This suggests D1_i is something we can multiply with φ(Q_i)

            # Since φ(Q_i) is [num_features] (row vector), D1_i should allow this multiplication
            # If the result should be [head_dim], then D1_i must be [num_features, head_dim]

            # But the paper says D1 is a matrix with rows being vec(...) ...
            # I think there's reshaping involved

            # Let me just implement it straightforward way:
            # D1[i, :, :] = Σ_j C[i,j] * (k_prime[j, :]^T @ v[j, :])
            # where k_prime[j, :] is [num_features] and v[j, :] is [head_dim]
            # So k_prime[j,:]^T @ v[j, :] is [num_features, 1] @ [1, head_dim] = [num_features, head_dim]

            # So D1[i] is [num_features, head_dim]
            # And D1 overall is [n, num_features, head_dim]

            # No wait, after applying Toeplitz, we get back to n positions
            # So D1 is [B, heads, n, num_features, head_dim]

            # Let me return this for now:
            return D1  # [B, heads, n, num_features, head_dim]

        else:
            # Compute D2 = C @ K'^T
            # This is used for the attention denominator

            # K'^T: [B, heads, num_features, n]
            k_prime_t = k_prime.transpose(2, 3)

            # For each position i: D2[i] = Σ_j C[i,j] K'[j]^T
            # K'[j]^T is [num_features] vector
            # So D2[i] is [num_features] vector
            # Overall D2 is [n, num_features]

            # Apply Toeplitz for each head and batch
            # Apply Toeplitz for each head (vectorized over batch)
            results = []
            for h in range(H):
                # k_prime_t[:, h]: [B, num_features, n]
                # Transpose: [B, num_features, n] -> [B, n, num_features]
                k_t_h = k_prime_t[:, h].transpose(1, 2)  # [B, n, num_features]

                # Vectorized: processes all B batches in parallel
                D2_h = fft_toeplitz_matmul(c[h], k_t_h)  # [B, n, num_features]
                results.append(D2_h)

            # Stack: [B, heads, n, num_features]
            D2 = torch.stack(results, dim=1)  # [B, heads, n, num_features]

            return D2  # [B, heads, n, num_features]

    def extra_repr(self) -> str:
        """String representation with KERPLE parameters."""
        return (
            f'num_patches={self.num_patches}, '
            f'dim={self.dim}, '
            f'heads={self.heads}, '
            f'max_rel_pos={self.max_rel_pos}, '
            f'type=KERPLE (FFT-based)'
        )
