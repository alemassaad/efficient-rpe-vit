"""
FFT-based Toeplitz matrix multiplication utilities.

This module implements O(n log n) Toeplitz matrix-vector multiplication
using Fast Fourier Transform, as described in Luo et al., NeurIPS 2021.

Reference:
    Luo et al., "Stable, Fast and Accurate: Kernelized Attention with
    Relative Positional Encoding", NeurIPS 2021, Section 3.2
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def fft_toeplitz_matmul(
    c: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Multiply Toeplitz matrix (defined by c) with matrix x using FFT.

    This achieves O(n log n) complexity instead of naive O(n²) by:
    1. Embedding the Toeplitz matrix in a circulant matrix (size 2n-1)
    2. Using FFT to compute the circulant matrix-vector product
    3. Extracting the relevant n elements

    Mathematical Background:
        A Toeplitz matrix has constant diagonals:
            T[i,j] = c[j-i]

        Such matrices can be embedded in circulant matrices, which are
        diagonalized by the DFT. This allows FFT-based multiplication.

    Args:
        c: Toeplitz coefficients of shape [2n-1] or [B, heads, 2n-1]
           Ordered as [c_{-(n-1)}, ..., c_{-1}, c_0, c_1, ..., c_{n-1}]
           where c_k represents the value at relative position k
        x: Matrix to multiply of shape [n, d] or [B, heads, n, d]

    Returns:
        Result of Toeplitz @ x, same shape as x

    Complexity:
        O(n log n * d) where n is sequence length, d is feature dimension

    Example:
        >>> # Create Toeplitz coefficients for sequence length 4
        >>> # c = [c_{-3}, c_{-2}, c_{-1}, c_0, c_1, c_2, c_3]
        >>> c = torch.randn(7)  # 2*4-1 = 7
        >>> x = torch.randn(4, 8)  # [n, d]
        >>> result = fft_toeplitz_matmul(c, x)  # [4, 8]
    """
    # Handle different input shapes
    if c.dim() == 1:
        # Single Toeplitz matrix: [2n-1]
        if x.dim() == 3:
            # Batched x: [B, n, d] - use vectorized batched implementation
            return _fft_toeplitz_matmul_batched(c, x)
        else:
            # Non-batched x: [n, d] or [n]
            return _fft_toeplitz_matmul_single(c, x)
    elif c.dim() == 3:
        # Batched multi-head: [B, heads, 2n-1]
        B, H, _ = c.shape
        if x.dim() == 4:
            # x is [B, heads, n, d]
            B_x, H_x, N, D = x.shape
            assert B == B_x and H == H_x, "Batch and head dimensions must match"

            # Process each head separately
            results = []
            for h in range(H):
                # For this head: c[h] is [2n-1], x[:, h] is [B, n, d]
                result_h = _fft_toeplitz_matmul_batched(c[:, h], x[:, h])
                results.append(result_h)

            # Stack back: [B, heads, n, d]
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"When c has 3 dims, x must have 4 dims. Got x.shape={x.shape}")
    else:
        raise ValueError(f"c must have 1 or 3 dimensions. Got shape={c.shape}")


def _fft_toeplitz_matmul_single(
    c: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Single Toeplitz matrix multiplication.

    Args:
        c: Toeplitz coefficients [2n-1]
        x: Matrix [n, d]

    Returns:
        Result [n, d]
    """
    if x.dim() == 1:
        # Handle vector case
        x = x.unsqueeze(-1)
        result = _fft_toeplitz_matmul_impl(c, x)
        return result.squeeze(-1)
    elif x.dim() == 2:
        return _fft_toeplitz_matmul_impl(c, x)
    else:
        raise ValueError(f"x must have 1 or 2 dimensions. Got shape={x.shape}")


def _fft_toeplitz_matmul_batched(
    c: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized batched Toeplitz matrix multiplication.

    This function processes all batch elements in parallel using vectorized FFT operations,
    eliminating the need for Python loops over the batch dimension.

    Args:
        c: Toeplitz coefficients [B, 2n-1] or [2n-1]
        x: Matrix [B, n, d]

    Returns:
        Result [B, n, d]
    """
    if c.dim() == 1:
        # Single coefficient tensor shared across batch
        # Expand to batch dimension
        c = c.unsqueeze(0).expand(x.shape[0], -1)

    B, N, D = x.shape
    n_coeffs = c.shape[1]  # 2n-1
    n = (n_coeffs + 1) // 2  # Sequence length

    assert N == n, f"Matrix height {N} doesn't match expected {n} from coefficients {n_coeffs}"

    # Build circulant first column for all batch elements in parallel
    # c: [B, 2n-1] -> circulant_col: [B, 2n-1]
    circulant_col = torch.cat([
        c[:, n-1:n],                          # [B, 1] - c_0
        torch.flip(c[:, :n-1], dims=[1]),     # [B, n-1] - negative indices reversed
        torch.flip(c[:, n:], dims=[1])        # [B, n-1] - positive indices reversed
    ], dim=1)

    # Compute FFT of circulant column for all batches at once
    # [B, 2n-1] -> [B, 2n-1]
    c_fft = torch.fft.fft(circulant_col, dim=-1)

    # Zero-pad x to length 2n-1 for all batch elements
    # [B, n, d] -> [B, 2n-1, d]
    x_padded = F.pad(x, (0, 0, 0, n - 1), mode='constant', value=0)

    # Compute FFT of padded x for all columns and batches simultaneously
    # [B, 2n-1, d] -> [B, 2n-1, d]
    x_fft = torch.fft.fft(x_padded, dim=1)

    # Element-wise multiply in frequency domain
    # c_fft: [B, 2n-1, 1] * x_fft: [B, 2n-1, d] -> [B, 2n-1, d]
    y_fft = c_fft.unsqueeze(-1) * x_fft

    # Compute IFFT for all batches and columns
    # [B, 2n-1, d] -> [B, 2n-1, d]
    y = torch.fft.ifft(y_fft, dim=1)

    # Extract first n elements (real part) for all batches
    # [B, 2n-1, d] -> [B, n, d]
    y_real = y[:, :n, :].real

    return y_real


def _fft_toeplitz_matmul_impl(
    c: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Core FFT-based Toeplitz multiplication implementation.

    Algorithm:
        1. Embed Toeplitz in circulant matrix
        2. Compute FFT of first column of circulant
        3. For each column of x:
            a. Zero-pad x to length 2n-1
            b. Compute FFT of padded x
            c. Element-wise multiply in frequency domain
            d. Compute IFFT
            e. Extract first n elements (real part)

    Args:
        c: Toeplitz coefficients [2n-1]
           Ordered as [c_{-(n-1)}, ..., c_{-1}, c_0, c_1, ..., c_{n-1}]
        x: Matrix [n, d]

    Returns:
        Result [n, d]

    Reference:
        Gray, R. M. (2006). Toeplitz and circulant matrices: A review.
    """
    # Get dimensions
    n_coeffs = c.shape[0]  # 2n-1
    n = (n_coeffs + 1) // 2  # Sequence length
    N, D = x.shape

    assert N == n, f"Matrix height {N} doesn't match expected {n} from coefficients {n_coeffs}"

    # The Toeplitz matrix T has:
    # T[i,j] = c[j - i + (n-1)]
    # where c is indexed as [c_{-(n-1)}, ..., c_{-1}, c_0, c_1, ..., c_{n-1}]
    #
    # For FFT-based multiplication using circular convolution, we need C[i,j] = col[(i-j) % m]
    # To embed Toeplitz in circulant, the first column must be:
    # [c_0, c_{-1}, c_{-2}, ..., c_{-(n-1)}, c_{n-1}, c_{n-2}, ..., c_1]
    #
    # In terms of the c array indices (where c[n-1] = c_0):
    # [c[n-1], c[n-2], ..., c[1], c[0], c[2n-2], c[2n-3], ..., c[n]]
    #
    # This requires reversing both the negative and positive index portions:
    # col = [c[n-1]] + reverse(c[0:n-1]) + reverse(c[n:2n-1])

    # Build circulant first column for FFT (requires flipping both parts)
    circulant_col = torch.cat([
        c[n-1:n],                        # [c_0]
        torch.flip(c[:n-1], dims=[0]),   # [c_{-1}, ..., c_{-(n-1)}] in reverse
        torch.flip(c[n:], dims=[0])      # [c_{n-1}, ..., c_1] in reverse
    ])

    # Compute FFT of circulant column (do once for all columns)
    # FFT length is 2n-1
    c_fft = torch.fft.fft(circulant_col)

    # Process each column of x
    results = []
    for d in range(D):
        # Zero-pad x column to length 2n-1
        x_col = x[:, d]  # [n]
        x_padded = F.pad(x_col, (0, n - 1), mode='constant', value=0)  # [2n-1]

        # Compute FFT of padded x
        x_fft = torch.fft.fft(x_padded)

        # Element-wise multiply in frequency domain
        y_fft = c_fft * x_fft

        # Compute IFFT
        y = torch.fft.ifft(y_fft)

        # Extract first n elements (real part)
        # Note: Result should be real, but take real() to handle numerical errors
        y_real = y[:n].real

        results.append(y_real)

    # Stack columns: [n, d]
    return torch.stack(results, dim=1)


def create_toeplitz_matrix(c: torch.Tensor, n: int) -> torch.Tensor:
    """
    Create explicit Toeplitz matrix from coefficients (for testing/debugging).

    This is the naive O(n²) approach, used only for validation.

    Args:
        c: Toeplitz coefficients [2n-1]
           Ordered as [c_{-(n-1)}, ..., c_{-1}, c_0, c_1, ..., c_{n-1}]
        n: Sequence length

    Returns:
        Toeplitz matrix [n, n] where T[i,j] = c[j-i + (n-1)]

    Example:
        >>> c = torch.tensor([4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])  # n=4
        >>> T = create_toeplitz_matrix(c, 4)
        >>> # T = [[1, 2, 3, 4],
        >>> #      [2, 1, 2, 3],
        >>> #      [3, 2, 1, 2],
        >>> #      [4, 3, 2, 1]]
    """
    T = torch.zeros(n, n, dtype=c.dtype, device=c.device)

    for i in range(n):
        for j in range(n):
            # Relative position: j - i
            # Index in c: (j - i) + (n - 1)
            idx = (j - i) + (n - 1)
            T[i, j] = c[idx]

    return T


def naive_toeplitz_matmul(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Naive O(n²) Toeplitz matrix multiplication (for testing).

    Args:
        c: Toeplitz coefficients [2n-1]
        x: Matrix [n, d]

    Returns:
        Result [n, d]
    """
    n = x.shape[0]
    T = create_toeplitz_matrix(c, n)
    return T @ x
