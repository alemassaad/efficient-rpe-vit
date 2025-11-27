# Circulant-STRING Implementation Guide

**Paper**: "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING" (Schenck et al., 2025)
**arXiv**: [2502.02562](https://arxiv.org/abs/2502.02562)

This document contains all mathematical details necessary to implement Circulant-STRING for our Performer-based Vision Transformers.

---

## 1. Overview

STRING (Separable Translationally Invariant Position Encodings) is a generalization of RoPE that:
- Maintains **translational invariance**: attention depends only on relative positions
- Is **separable**: transforms queries and keys independently
- Supports **arbitrary coordinate dimensionality** (1D, 2D, 3D, etc.)
- Can be computed efficiently via **FFT in O(d log d)** time

**Circulant-STRING** is a specific parameterization using circulant matrices that enables efficient FFT-based computation.

---

## 2. Core Mathematical Framework

### 2.1 STRING Definition

The position encoding is defined as a mapping from position vectors to rotation matrices:

$$\mathbf{R}(\mathbf{r}_i) = \exp\left(\sum_{k=1}^{d_c} \mathbf{L}_k [\mathbf{r}_i]_k\right)$$

Where:
- $\mathbf{R}: \mathbb{R}^{d_c} \rightarrow \mathbb{R}^{d \times d}$ maps position vectors to transformation matrices
- $d_c$ = coordinate dimensionality (2 for images with x,y positions)
- $d$ = embedding/token dimension
- $[\mathbf{r}_i]_k$ = the k-th coordinate of position vector $\mathbf{r}_i$
- $\{\mathbf{L}_k\}_{k=1}^{d_c}$ = learnable, commuting, skew-symmetric generators
- $\exp(\cdot)$ = matrix exponential: $\sum_{i=0}^{\infty} \frac{\mathbf{A}^i}{i!}$

### 2.2 Generator Requirements

The generators must satisfy:

1. **Skew-symmetry**: $\mathbf{L}_k^T = -\mathbf{L}_k$ for all $k$
2. **Commutativity**: $[\mathbf{L}_i, \mathbf{L}_j] = \mathbf{L}_i\mathbf{L}_j - \mathbf{L}_j\mathbf{L}_i = 0$ for all $i, j$

These ensure:
- $\mathbf{R}(\mathbf{r}_i) \in SO(d)$ (orthogonal matrix with det = 1)
- $\mathbf{R}(\mathbf{r}_i)^T \mathbf{R}(\mathbf{r}_j) = \mathbf{R}(\mathbf{r}_j - \mathbf{r}_i)$ (translational invariance)

---

## 3. Circulant Matrix Parameterization

### 3.1 Circulant Matrix Structure

A circulant matrix $\mathbf{C}$ is defined by its first row $[c_0, c_1, \ldots, c_{d-1}]$:

$$\mathbf{C} = \begin{pmatrix}
c_0 & c_{d-1} & c_{d-2} & \cdots & c_1 \\
c_1 & c_0 & c_{d-1} & \cdots & c_2 \\
c_2 & c_1 & c_0 & \cdots & c_3 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_{d-1} & c_{d-2} & c_{d-3} & \cdots & c_0
\end{pmatrix}$$

Each row is the previous row shifted (rotated) by one position to the right.

### 3.2 Key Properties of Circulant Matrices

1. **Circulant matrices commute**: $\mathbf{C}_1 \mathbf{C}_2 = \mathbf{C}_2 \mathbf{C}_1$
2. **Sum of circulant matrices is circulant**
3. **Transpose of circulant is circulant**: $\mathbf{C}^T$ has first row $[c_0, c_1, c_{d-1}, \ldots, c_2]$
4. **Diagonalized by DFT matrix**: $\mathbf{C} = \mathbf{F}^{-1} \text{diag}(\boldsymbol{\lambda}) \mathbf{F}$

### 3.3 Constructing Skew-Symmetric Generators

For Circulant-STRING, we define:

$$\mathbf{L}_k = \mathbf{C}_k - \mathbf{C}_k^T$$

Where $\mathbf{C}_k$ is a learnable circulant matrix with parameters $\{c_0^{(k)}, c_1^{(k)}, \ldots, c_{d-1}^{(k)}\}$.

**Why this works**:
- $\mathbf{L}_k^T = (\mathbf{C}_k - \mathbf{C}_k^T)^T = \mathbf{C}_k^T - \mathbf{C}_k = -\mathbf{L}_k$ ✓ (skew-symmetric)
- Circulant matrices commute, so $\mathbf{L}_k$ matrices also commute ✓

---

## 4. FFT-Based Efficient Computation (Theorem 3.5)

### 4.1 Eigenvalue Decomposition of Circulant Matrices

Every circulant matrix can be factorized as:

$$\mathbf{C} = \mathbf{F}^{-1} \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{d-1}) \mathbf{F}$$

Where:
- $\mathbf{F}$ is the DFT (Discrete Fourier Transform) matrix
- Eigenvalues are computed via FFT of the first row:

$$\lambda_j = \sum_{m=0}^{d-1} c_m \exp\left(-\frac{2\pi i \cdot j \cdot m}{d}\right) = \text{FFT}([c_0, c_1, \ldots, c_{d-1}])_j$$

### 4.2 Eigenvalues of Skew-Symmetric Circulant

For $\mathbf{L} = \mathbf{C} - \mathbf{C}^T$:

$$\boldsymbol{\mu} = \text{FFT}(\mathbf{c}) - \text{FFT}(\mathbf{c})^*$$

Where $\mathbf{c} = [c_0, c_1, \ldots, c_{d-1}]$ and $^*$ denotes complex conjugate.

**Note**: Since $\mathbf{L}$ is real and skew-symmetric, its eigenvalues are purely imaginary.

### 4.3 Matrix Exponential via FFT

Using spectral decomposition:

$$\exp(r \cdot \mathbf{L}) = \mathbf{F}^{-1} \text{diag}(\exp(r \cdot \mu_0), \ldots, \exp(r \cdot \mu_{d-1})) \mathbf{F}$$

**Algorithm** for applying $\exp(\mathbf{L}) \mathbf{z}$:

```
Input: circulant coefficients c = [c_0, ..., c_{d-1}], position scalar r, token z
Output: position-encoded token z'

1. λ_C = FFT(c)                           # Eigenvalues of C
2. λ_CT = conj(FFT(c))                    # Eigenvalues of C^T (complex conjugate)
3. μ = λ_C - λ_CT                         # Eigenvalues of L = C - C^T (purely imaginary)
4. μ_scaled = r * μ                       # Scale by position coordinate
5. exp_μ = exp(μ_scaled)                  # Element-wise exponential
6. z_freq = FFT(z)                        # Transform token to frequency domain
7. z_freq_scaled = exp_μ * z_freq         # Element-wise multiplication
8. z' = IFFT(z_freq_scaled)               # Transform back to spatial domain

Complexity: O(d log d) time, O(d) memory
```

### 4.4 Multi-Coordinate Extension (for 2D images)

For $d_c$ coordinates (e.g., $d_c = 2$ for x,y positions):

$$\mathbf{R}(\mathbf{r}_i) = \exp\left(\sum_{k=1}^{d_c} [\mathbf{r}_i]_k \cdot \mathbf{L}_k\right)$$

Since generators commute:

$$\exp\left(\sum_k r_k \mathbf{L}_k\right) = \prod_k \exp(r_k \mathbf{L}_k)$$

**Algorithm for 2D positions**:

```
Input:
  - c_x = [c_0^x, ..., c_{d-1}^x]  # Learnable coefficients for x-axis
  - c_y = [c_0^y, ..., c_{d-1}^y]  # Learnable coefficients for y-axis
  - r_i = (x_i, y_i)               # 2D position of token i
  - z_i                            # Token embedding

1. Compute eigenvalues for both axes:
   μ_x = FFT(c_x) - conj(FFT(c_x))
   μ_y = FFT(c_y) - conj(FFT(c_y))

2. Combine scaled eigenvalues:
   μ_combined = x_i * μ_x + y_i * μ_y

3. Apply via FFT:
   z_freq = FFT(z_i)
   z_freq_encoded = exp(μ_combined) * z_freq
   z'_i = IFFT(z_freq_encoded)
```

---

## 5. Application to Attention

### 5.1 Transforming Queries and Keys

Given queries $\mathbf{q}_i$ and keys $\mathbf{k}_j$ at positions $\mathbf{r}_i$ and $\mathbf{r}_j$:

$$\mathbf{q}'_i = \mathbf{R}(\mathbf{r}_i) \mathbf{q}_i$$
$$\mathbf{k}'_j = \mathbf{R}(\mathbf{r}_j) \mathbf{k}_j$$

### 5.2 Attention Score Computation

Standard softmax attention:
$$\text{Attention}_{ij} = \text{softmax}\left(\mathbf{q}'^T_i \mathbf{k}'_j\right)$$

**Key property** (translational invariance):
$$\mathbf{q}'^T_i \mathbf{k}'_j = \mathbf{q}_i^T \mathbf{R}(\mathbf{r}_i)^T \mathbf{R}(\mathbf{r}_j) \mathbf{k}_j = \mathbf{q}_i^T \mathbf{R}(\mathbf{r}_j - \mathbf{r}_i) \mathbf{k}_j$$

The attention depends only on the **relative position** $\mathbf{r}_j - \mathbf{r}_i$.

### 5.3 Values Are NOT Transformed

According to the paper, only queries and keys receive position encoding. Values $\mathbf{v}$ remain unchanged:

$$\text{Output}_i = \sum_j \text{Attention}_{ij} \cdot \mathbf{v}_j$$

---

## 6. Integration with Linear Attention (Performers)

For Performer-style linear attention with feature map $\phi$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \frac{\phi(\mathbf{Q}') (\phi(\mathbf{K}')^T \mathbf{V})}{\phi(\mathbf{Q}') \phi(\mathbf{K}')^T \mathbf{1}}$$

Where $\mathbf{Q}' = \mathbf{R}(\mathbf{r}) \mathbf{Q}$ and $\mathbf{K}' = \mathbf{R}(\mathbf{r}) \mathbf{K}$.

**Note from paper**: "Separability also makes RoPE [and STRING] compatible with linear attention, e.g. Performers. Here, the attention matrix is not instantiated in memory so explicit RPE mechanisms are not possible."

STRING/Circulant-STRING is explicitly designed to work with Performers!

---

## 7. Learnable Parameters

### 7.1 Parameter Count

For Circulant-STRING with:
- Embedding dimension $d$
- Coordinate dimensionality $d_c$ (e.g., 2 for images)
- $H$ attention heads

**Per-head parameters**: $d_c \times d$ scalars (circulant coefficients)
**Total parameters**: $H \times d_c \times d$

For ViT-Base ($d=768$, $H=12$, $d_c=2$):
- Total: $12 \times 2 \times 768 = 18,432$ parameters (negligible compared to model size)

### 7.2 Per-Head vs Shared

From the paper: "the orthogonal matrix $\mathbf{P}$ is independent of the coordinates...so it can be learned and stored once per attention head and shared across all training examples."

This indicates **per-head** learnable parameters.

### 7.3 Block Size Optimization

The paper mentions: "For Circulant-STRING, we swept over block sizes in the set {4, 8, 16, 32, 64} to find the optimal setting (often around 16)."

This suggests using **block-circulant** structure for efficiency, where $d$ is divided into blocks.

---

## 8. Parameter Initialization

### 8.1 Recommended Initialization

Initialize near identity (small rotations):

```python
# Option 1: Zero initialization (identity at start)
c_0 = 0.0
c_j = 0.0 for j > 0

# Option 2: Small random initialization
c_0 = 0.0  # Keep diagonal stable
c_j ~ N(0, 1/d) for j > 0  # Small off-diagonal

# Option 3: Xavier/Glorot style
c_j ~ Uniform(-sqrt(3/d), sqrt(3/d))
```

### 8.2 Rationale

- $c_0$ on diagonal: setting $c_0 = 0$ keeps the initial transformation close to identity
- Small off-diagonal: prevents large initial rotations that could destabilize training

---

## 9. Numerical Stability

### 9.1 No Overflow Risk

**Important**: The eigenvalues of L = C - C^T are **purely imaginary** (μ = iθ).

For purely imaginary μ:
- exp(iθ) = cos(θ) + i·sin(θ)
- This lies on the unit circle with |exp(iθ)| = 1
- **There is no overflow risk!**

The original documentation incorrectly suggested clipping `mu.real`. Since μ.real ≈ 0 always, this is unnecessary.

Optional: Clip the imaginary part to prevent very rapid rotations:
```python
# Optional: limit rotation angles (rarely needed)
mu_imag_clipped = torch.clamp(mu.imag, min=-100, max=100)
exp_mu = torch.exp(1j * mu_imag_clipped)
```

### 9.2 FFT Precision

Use double precision for FFT operations:

```python
# Compute in float64, then convert back
c_double = c.double()
lambda_c = torch.fft.fft(c_double)
# ... computation ...
result = result.to(original_dtype)
```

### 9.3 Gradient Flow

The matrix exponential has stable gradients for small arguments. Add regularization:

$$\mathcal{L}_{\text{reg}} = \lambda_{\text{reg}} \sum_{k,j} (c_j^{(k)})^2$$

Recommended: $\lambda_{\text{reg}} = 10^{-6}$

---

## 10. Hyperparameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| Learning rate for circulant params | $10^{-4}$ | $[10^{-5}, 10^{-3}]$ | Can use same as model LR |
| Weight decay | $10^{-5}$ | $[10^{-6}, 10^{-4}]$ | L2 regularization |
| Gradient clipping | 5.0 | $[1.0, 10.0]$ | Prevents instability |
| Block size | 16 | $\{4, 8, 16, 32, 64\}$ | Tune per task |
| Max wavelength (for position normalization) | 100 | - | From RoPE-Mixed |

---

## 11. Implementation Pseudocode

### 11.1 CirculantSTRING Module

```python
class CirculantSTRING(nn.Module):
    def __init__(self, dim, num_heads, coord_dim=2):
        """
        Args:
            dim: embedding dimension (d)
            num_heads: number of attention heads (H)
            coord_dim: coordinate dimensionality (d_c), 2 for images
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.coord_dim = coord_dim
        self.head_dim = dim // num_heads

        # Learnable circulant coefficients: (num_heads, coord_dim, head_dim)
        self.circulant_coeffs = nn.Parameter(
            torch.zeros(num_heads, coord_dim, self.head_dim)
        )

        # Small initialization for non-zero elements
        nn.init.normal_(self.circulant_coeffs, mean=0, std=0.01)

    def get_eigenvalues(self, coeffs):
        """
        Compute eigenvalues of L = C - C^T from circulant coefficients.

        Args:
            coeffs: (..., d) circulant first-row coefficients
        Returns:
            eigenvalues: (..., d) complex eigenvalues (purely imaginary)
        """
        # FFT of coefficients gives eigenvalues of C
        lambda_c = torch.fft.fft(coeffs, dim=-1)

        # Eigenvalues of C^T are complex conjugates
        # Eigenvalues of L = C - C^T
        eigenvalues = lambda_c - torch.conj(lambda_c)

        return eigenvalues  # Purely imaginary

    def apply_rotation(self, x, positions):
        """
        Apply position-dependent rotation to queries or keys.

        Args:
            x: (batch, seq_len, num_heads, head_dim) queries or keys
            positions: (batch, seq_len, coord_dim) position coordinates
        Returns:
            x_rotated: (batch, seq_len, num_heads, head_dim)
        """
        B, N, H, D = x.shape

        # Get eigenvalues for each head and coordinate
        # eigenvalues: (num_heads, coord_dim, head_dim)
        eigenvalues = self.get_eigenvalues(self.circulant_coeffs)

        # Scale eigenvalues by positions and sum across coordinates
        # positions: (B, N, coord_dim) -> (B, N, 1, coord_dim, 1)
        pos = positions.unsqueeze(2).unsqueeze(-1)

        # eigenvalues: (1, 1, H, coord_dim, D)
        eig = eigenvalues.unsqueeze(0).unsqueeze(0)

        # Weighted sum: (B, N, H, D)
        mu_combined = (pos * eig).sum(dim=-2)

        # Apply rotation via FFT
        x_freq = torch.fft.fft(x.to(torch.complex64), dim=-1)
        x_freq_rotated = torch.exp(mu_combined) * x_freq
        x_rotated = torch.fft.ifft(x_freq_rotated, dim=-1).real

        return x_rotated.to(x.dtype)

    def forward(self, q, k, positions):
        """
        Apply Circulant-STRING to queries and keys.

        Args:
            q: (batch, seq_len, num_heads, head_dim) queries
            k: (batch, seq_len, num_heads, head_dim) keys
            positions: (batch, seq_len, coord_dim) 2D positions
        Returns:
            q_encoded: (batch, seq_len, num_heads, head_dim)
            k_encoded: (batch, seq_len, num_heads, head_dim)
        """
        q_encoded = self.apply_rotation(q, positions)
        k_encoded = self.apply_rotation(k, positions)

        return q_encoded, k_encoded
```

### 11.2 Position Computation for ViT Patches

```python
def get_2d_positions(num_patches_h, num_patches_w, normalize=True):
    """
    Generate 2D position coordinates for image patches.

    Args:
        num_patches_h: number of patches in height
        num_patches_w: number of patches in width
        normalize: whether to normalize positions
    Returns:
        positions: (num_patches, 2) tensor of (x, y) coordinates
    """
    y_coords = torch.arange(num_patches_h).float()
    x_coords = torch.arange(num_patches_w).float()

    # Create grid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    if normalize:
        # Normalize to [-1, 1] or [0, 1]
        positions[:, 0] /= (num_patches_w - 1)
        positions[:, 1] /= (num_patches_h - 1)

    return positions
```

---

## 12. Comparison with Other Methods

| Method | Time Complexity | Space Complexity | Learnable | Translational Invariance |
|--------|-----------------|------------------|-----------|--------------------------|
| Absolute PE | O(N·d) | O(N·d) | Yes | No |
| RoPE | O(N·d) | O(d) | Partial (frequencies) | Yes |
| Circulant-STRING | O(N·d·log d) | O(d) | Yes (full) | Yes |
| Dense STRING | O(N·d³) | O(N·d²) | Yes | Yes |

---

## 13. Key Differences from KERPLE

| Aspect | KERPLE | Circulant-STRING |
|--------|--------|------------------|
| Bias type | Additive to attention logits | Multiplicative rotation of Q/K |
| Computation | Toeplitz FFT convolution | Matrix exponential via FFT |
| Parameters | Distance kernel parameters | Circulant matrix coefficients |
| With Performer | Modifies feature map products | Rotates Q/K before feature map |
| Translational invariance | Via distance function | Via Lie group structure |

---

## 14. Training Considerations

### 14.1 Transfer from 2D Pre-trained Models

From the paper: "STRING can be trained from regular 2D pre-trained checkpoint" by initializing $\mathbf{L}_k$ near zero.

This means we can:
1. Start from a pre-trained ViT
2. Initialize Circulant-STRING parameters to zero
3. Fine-tune to learn the position encoding

### 14.2 Batch Processing

Stack eigenvalue computations via batched FFT for efficiency:

```python
# Process all positions in batch
positions = positions.view(B * N, coord_dim)  # Flatten
# ... batch computation ...
result = result.view(B, N, H, D)  # Unflatten
```

---

## 15. Expected Benefits for Our Project

1. **Performer Compatibility**: Explicitly designed for linear attention
2. **2D Position Handling**: Native support for image patch positions
3. **Computational Efficiency**: O(d log d) per token
4. **State-of-the-Art Results**: Outperforms RoPE in the paper's experiments
5. **Small Parameter Overhead**: ~18K parameters for ViT-Base

---

## References

1. Schenck et al. (2025). "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING". arXiv:2502.02562
2. Su et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding". Neurocomputing.
3. Choromanski et al. (2020). "Rethinking Attention with Performers". arXiv:2009.14794
