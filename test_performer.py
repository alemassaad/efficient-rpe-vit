"""
Comprehensive tests for FAVOR+ Performer implementation.

This test suite validates:
1. Random feature generation (orthogonal and i.i.d.)
2. Positive feature map computation
3. FAVOR+ attention forward/backward pass
4. PerformerViT model functionality
5. Approximation quality compared to standard attention
"""

import math
import torch
import torch.nn as nn
import unittest

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.attention.favor_plus import FAVORPlusAttention
from models import create_model
from configs.datasets.mnist import MNIST_CONFIG
from configs.datasets.cifar10 import CIFAR10_CONFIG


class TestFAVORPlusAttention(unittest.TestCase):
    """Test FAVOR+ attention mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.heads = 4
        self.seq_len = 16
        self.batch_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_orthogonal_features(self):
        """Test that orthogonal random features are properly orthogonal."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        # Get random features
        omega = attention.omega  # Shape: (heads, dim_head, num_features)

        for h in range(self.heads):
            # Get features for this head
            features = omega[h]  # (dim_head, num_features)

            # Check orthogonality: features^T @ features should be close to scaled identity
            # when num_features <= dim_head
            if features.shape[1] <= features.shape[0]:
                product = features.T @ features
                expected = torch.eye(features.shape[1], device=self.device) * features.shape[0]

                # Check if close to scaled identity (with reasonable tolerance)
                self.assertTrue(
                    torch.allclose(product, expected, rtol=1e-4, atol=1e-4),
                    f"Random features for head {h} are not orthogonal"
                )

    def test_iid_gaussian_features(self):
        """Test that i.i.d. Gaussian features have correct statistics."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=False
        ).to(self.device)

        omega = attention.omega  # Shape: (heads, dim_head, num_features)

        # Flatten to check statistics
        flat_omega = omega.flatten()

        # Check mean is close to 0
        mean = flat_omega.mean().item()
        self.assertAlmostEqual(mean, 0.0, places=1,
                              msg=f"Mean {mean} not close to 0")

        # Check std is close to 1
        std = flat_omega.std().item()
        self.assertAlmostEqual(std, 1.0, places=1,
                              msg=f"Std {std} not close to 1")

    def test_positive_features(self):
        """Test that positive feature map produces only positive values."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)

        # Process through attention to get Q, K after projection
        with torch.no_grad():
            qkv = attention.qkv(x).chunk(3, dim=-1)
            q, k, _ = map(
                lambda t: t.reshape(self.batch_size, self.seq_len,
                                   self.heads, self.dim // self.heads).transpose(1, 2),
                qkv
            )

            # Apply scaling
            q = q * attention.scale
            k = k * attention.scale

            # Apply positive features
            q_prime = attention._phi_positive(q)
            k_prime = attention._phi_positive(k)

        # Check all values are positive
        self.assertTrue((q_prime >= 0).all(),
                       "Q features contain negative values")
        self.assertTrue((k_prime >= 0).all(),
                       "K features contain negative values")

    def test_forward_pass(self):
        """Test FAVOR+ attention forward pass."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)

        # Forward pass
        output = attention(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape,
                        f"Output shape {output.shape} != input shape {x.shape}")

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any(),
                        "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(),
                        "Output contains Inf values")

    def test_gradient_flow(self):
        """Test that gradients flow through FAVOR+ attention."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        # Create input with gradient tracking
        x = torch.randn(self.batch_size, self.seq_len, self.dim,
                       requires_grad=True).to(self.device)

        # Forward pass
        output = attention(x)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist and are not zero
        self.assertIsNotNone(x.grad, "Input gradient is None")
        self.assertFalse((x.grad == 0).all(),
                        "All input gradients are zero")

        # Check parameter gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad,
                                    f"Gradient for {name} is None")
                self.assertFalse((param.grad == 0).all(),
                               f"All gradients for {name} are zero")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        # Test with large values
        x_large = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device) * 10
        output_large = attention(x_large)
        self.assertFalse(torch.isnan(output_large).any(),
                        "NaN in output with large inputs")
        self.assertFalse(torch.isinf(output_large).any(),
                        "Inf in output with large inputs")

        # Test with small values
        x_small = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device) * 0.01
        output_small = attention(x_small)
        self.assertFalse(torch.isnan(output_small).any(),
                        "NaN in output with small inputs")


class TestPerformerTransformerBlock(unittest.TestCase):
    """Test Performer transformer block."""

    def test_forward_pass(self):
        """Test transformer block forward pass."""
        dim = 64
        heads = 4
        mlp_dim = 128
        seq_len = 16
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        block = PerformerTransformerBlock(
            dim=dim,
            heads=heads,
            mlp_dim=mlp_dim
        ).to(device)

        x = torch.randn(batch_size, seq_len, dim).to(device)
        output = block(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestPerformerViT(unittest.TestCase):
    """Test PerformerViT model."""

    def test_mnist_forward(self):
        """Test PerformerViT with MNIST input dimensions."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model with MNIST config
        model = create_performer_vit(MNIST_CONFIG).to(device)

        # Create dummy MNIST input
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28).to(device)

        # Forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 10),
                        f"Output shape {output.shape} != expected (4, 10)")

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cifar10_forward(self):
        """Test PerformerViT with CIFAR-10 input dimensions."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model with CIFAR-10 config
        model = create_performer_vit(CIFAR10_CONFIG).to(device)

        # Create dummy CIFAR-10 input
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32).to(device)

        # Forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 10),
                        f"Output shape {output.shape} != expected (4, 10)")

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        # MNIST model
        mnist_model = create_performer_vit(MNIST_CONFIG)
        mnist_params = mnist_model.count_parameters()

        # Should be close to baseline
        expected_mnist = MNIST_CONFIG['expected_params']
        actual_mnist = mnist_params['trainable']
        diff_pct = abs(actual_mnist - expected_mnist) / expected_mnist * 100

        self.assertLess(diff_pct, 20,
                       f"MNIST params {actual_mnist} differ by {diff_pct:.1f}% from expected {expected_mnist}")

        # Check random features are counted
        self.assertGreater(mnist_params['random_features'], 0,
                          "Random features not counted")

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_performer_vit(MNIST_CONFIG).to(device)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 1, 28, 28).to(device)
            output = model(x)
            self.assertEqual(output.shape[0], batch_size)


class TestApproximationQuality(unittest.TestCase):
    """Test FAVOR+ approximation quality compared to standard attention."""

    def test_approximation_error(self):
        """Test that FAVOR+ approximates standard attention reasonably well."""
        dim = 64
        heads = 4
        seq_len = 16
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create standard attention
        standard_attn = nn.MultiheadAttention(
            dim, heads, batch_first=True
        ).to(device)

        # Create FAVOR+ attention with more random features for better approximation
        favor_attn = FAVORPlusAttention(
            dim=dim,
            heads=heads,
            num_features=256  # More features for better approximation
        ).to(device)

        # Copy weights from standard to FAVOR+ for fair comparison
        with torch.no_grad():
            # Copy QKV weights
            favor_attn.qkv.weight.data = standard_attn.in_proj_weight.data
            # Note: FAVORPlusAttention qkv has no bias by design

            # Copy output projection weights
            favor_attn.out_proj.weight.data = standard_attn.out_proj.weight.data
            if standard_attn.out_proj.bias is not None and favor_attn.out_proj.bias is not None:
                favor_attn.out_proj.bias.data = standard_attn.out_proj.bias.data

        # Create input
        x = torch.randn(batch_size, seq_len, dim).to(device)

        # Forward pass through both
        with torch.no_grad():
            standard_output, _ = standard_attn(x, x, x)
            favor_output = favor_attn(x)

        # Compute relative error
        diff = (standard_output - favor_output).norm()
        relative_error = diff / standard_output.norm()

        # Should be reasonably close (not identical due to approximation)
        self.assertLess(relative_error, 0.5,
                       f"Relative error {relative_error:.3f} too large")

        print(f"\nApproximation quality: {relative_error:.3f} relative error")
        print(f"  Standard output norm: {standard_output.norm():.3f}")
        print(f"  FAVOR+ output norm: {favor_output.norm():.3f}")
        print(f"  Difference norm: {diff:.3f}")


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of FAVOR+ vs standard attention."""

    def test_memory_scaling(self):
        """Test that FAVOR+ uses less memory than standard attention."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        device = torch.device('cuda')
        dim = 64
        heads = 4
        batch_size = 2

        # Test with increasing sequence lengths
        seq_lengths = [32, 64, 128]
        favor_memory = []
        standard_memory = []

        for seq_len in seq_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # FAVOR+ attention
            favor_attn = FAVORPlusAttention(dim=dim, heads=heads).to(device)
            x = torch.randn(batch_size, seq_len, dim).to(device)
            _ = favor_attn(x)
            favor_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            favor_memory.append(favor_mem)

            del favor_attn, x
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Standard attention (using our baseline implementation)
            from models.components.attention import MultiHeadAttention
            standard_attn = MultiHeadAttention(dim=dim, heads=heads).to(device)
            x = torch.randn(batch_size, seq_len, dim).to(device)
            _ = standard_attn(x)
            standard_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            standard_memory.append(standard_mem)

            del standard_attn, x

            print(f"\nSeq length {seq_len}:")
            print(f"  FAVOR+ memory: {favor_mem:.2f} MB")
            print(f"  Standard memory: {standard_mem:.2f} MB")
            print(f"  Ratio: {standard_mem/favor_mem:.2f}x")

        # Check that FAVOR+ scales better (linear vs quadratic)
        # The ratio should increase with sequence length
        ratio_improvement = (standard_memory[-1] / favor_memory[-1]) / (standard_memory[0] / favor_memory[0])
        self.assertGreater(ratio_improvement, 1.2,
                          "FAVOR+ not showing better memory scaling")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFAVORPlusAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformerTransformerBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformerViT))
    suite.addTests(loader.loadTestsFromTestCase(TestApproximationQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryEfficiency))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("PERFORMER FAVOR+ IMPLEMENTATION TESTS")
    print("=" * 60)

    success = run_tests()

    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)