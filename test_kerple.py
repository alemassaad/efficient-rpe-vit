"""
Comprehensive test suite for KERPLE (Kernelized Attention with RPE).

Tests the implementation of Luo et al., NeurIPS 2021:
"Stable, Fast and Accurate: Kernelized Attention with Relative Positional Encoding"
"""

import unittest
import torch
import torch.nn as nn
import math
from models.rpe.fft_utils import (
    fft_toeplitz_matmul,
    naive_toeplitz_matmul,
    create_toeplitz_matrix
)
from models.rpe import KERPLEPositionalEncoding
from models.attention import FAVORPlusAttention, ReLUAttention, SoftmaxAttention
from models import create_model
from configs.datasets.mnist import MNISTConfig


class TestFFTToeplitzMultiplication(unittest.TestCase):
    """Test FFT-based Toeplitz matrix multiplication."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.n = 8  # Sequence length
        self.d = 4  # Feature dimension
        self.c = torch.randn(2 * self.n - 1)  # Toeplitz coefficients
        self.x = torch.randn(self.n, self.d)  # Matrix to multiply

    def test_correctness_vs_naive(self):
        """Verify FFT result matches naive O(n²) implementation."""
        # Compute using FFT (O(n log n))
        result_fft = fft_toeplitz_matmul(self.c, self.x)

        # Compute using naive method (O(n²))
        result_naive = naive_toeplitz_matmul(self.c, self.x)

        # Should be very close (within floating point error)
        self.assertTrue(
            torch.allclose(result_fft, result_naive, rtol=1e-4, atol=1e-5),
            f"FFT and naive results differ:\nFFT:\n{result_fft}\nNaive:\n{result_naive}"
        )

    def test_output_shape(self):
        """Verify output shape matches input."""
        result = fft_toeplitz_matmul(self.c, self.x)
        self.assertEqual(result.shape, self.x.shape)

    def test_batch_dimensions(self):
        """Test handling of batched inputs."""
        B = 2
        H = 3
        c_batched = torch.randn(B, H, 2 * self.n - 1)
        x_batched = torch.randn(B, H, self.n, self.d)

        result = fft_toeplitz_matmul(c_batched, x_batched)

        self.assertEqual(result.shape, x_batched.shape)

    def test_toeplitz_structure(self):
        """Verify Toeplitz matrix construction is correct."""
        T = create_toeplitz_matrix(self.c, self.n)

        # Check shape
        self.assertEqual(T.shape, (self.n, self.n))

        # Check Toeplitz property: T[i,j] = T[i+1,j+1] (constant diagonals)
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                self.assertAlmostEqual(
                    T[i, j].item(),
                    T[i + 1, j + 1].item(),
                    places=5
                )

    def test_gradient_flow(self):
        """Verify gradients backpropagate through FFT multiplication."""
        c = self.c.clone().requires_grad_(True)
        x = self.x.clone().requires_grad_(True)

        result = fft_toeplitz_matmul(c, x)
        loss = result.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        self.assertIsNotNone(c.grad)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(c.grad != 0))
        self.assertTrue(torch.any(x.grad != 0))


class TestKERPLEPositionalEncoding(unittest.TestCase):
    """Test KERPLE RPE module."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.num_patches = 16
        self.dim = 64
        self.heads = 4
        self.head_dim = self.dim // self.heads
        self.num_features = 32

        self.rpe = KERPLEPositionalEncoding(
            num_patches=self.num_patches,
            dim=self.dim,
            heads=self.heads
        )

        # Create sample kernelized features
        B = 2
        self.k_prime = torch.randn(B, self.heads, self.num_patches, self.num_features)
        self.v = torch.randn(B, self.heads, self.num_patches, self.head_dim)

    def test_parameter_initialization(self):
        """Verify learnable parameters are initialized correctly."""
        # Should have rel_pos_bias parameter
        self.assertTrue(hasattr(self.rpe, 'rel_pos_bias'))

        # Shape should be [heads, 2n-1]
        expected_shape = (self.heads, 2 * self.num_patches - 1)
        self.assertEqual(self.rpe.rel_pos_bias.shape, expected_shape)

        # Should be a parameter (trainable)
        self.assertIsInstance(self.rpe.rel_pos_bias, nn.Parameter)

    def test_apply_rpe_fft_with_v(self):
        """Test D1 = C @ (K'^T @ V) computation."""
        D1 = self.rpe.apply_rpe_fft(self.k_prime, self.v)

        # Check output shape
        B, H, N, F = self.k_prime.shape
        _, _, _, D = self.v.shape

        expected_shape = (B, H, N, F, D)
        self.assertEqual(D1.shape, expected_shape)

    def test_apply_rpe_fft_without_v(self):
        """Test D2 = C @ K'^T computation."""
        D2 = self.rpe.apply_rpe_fft(self.k_prime, None)

        # Check output shape
        B, H, N, F = self.k_prime.shape

        expected_shape = (B, H, N, F)
        self.assertEqual(D2.shape, expected_shape)

    def test_toeplitz_coefficients(self):
        """Verify c = exp(b) and correct shape."""
        # Access coefficients (used internally)
        c = torch.exp(self.rpe.rel_pos_bias)

        # Should be positive (since exp)
        self.assertTrue(torch.all(c > 0))

        # Shape should be [heads, 2n-1]
        expected_shape = (self.heads, 2 * self.num_patches - 1)
        self.assertEqual(c.shape, expected_shape)

    def test_gradient_backprop(self):
        """Verify gradients flow to rel_pos_bias."""
        # Zero gradients
        self.rpe.zero_grad()

        # Forward pass
        D1 = self.rpe.apply_rpe_fft(self.k_prime, self.v)

        # Backward pass
        loss = D1.sum()
        loss.backward()

        # Gradients should exist
        self.assertIsNotNone(self.rpe.rel_pos_bias.grad)
        # And be non-zero
        self.assertTrue(torch.any(self.rpe.rel_pos_bias.grad != 0))

    def test_forward_raises_error(self):
        """Verify forward() raises NotImplementedError."""
        x = torch.randn(2, self.num_patches, self.dim)

        with self.assertRaises(NotImplementedError):
            self.rpe(x)


class TestKERPLEIntegration(unittest.TestCase):
    """Test KERPLE integration with attention mechanisms."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.dim = 64
        self.heads = 4

        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)

        self.rpe = KERPLEPositionalEncoding(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        )

    def test_favor_plus_with_kerple(self):
        """Test FAVOR+ + KERPLE end-to-end."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        # Forward pass with RPE
        output = attention(self.x, rpe=self.rpe)

        # Check output shape
        self.assertEqual(output.shape, self.x.shape)

        # Check for NaN/Inf (training stability)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_relu_with_kerple(self):
        """Test ReLU + KERPLE end-to-end."""
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        # Forward pass with RPE
        output = attention(self.x, rpe=self.rpe)

        # Check output shape
        self.assertEqual(output.shape, self.x.shape)

        # Check for NaN/Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_qk_normalization_applied(self):
        """Verify Q/K normalization when RPE is used."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        # We can't directly check normalization without modifying the code,
        # but we can check that output is different with/without RPE
        output_with_rpe = attention(self.x, rpe=self.rpe)
        output_without_rpe = attention(self.x, rpe=None)

        # Outputs should be different (RPE changes computation)
        self.assertFalse(torch.allclose(output_with_rpe, output_without_rpe))

    def test_softmax_rejects_kerple(self):
        """Verify softmax attention raises error with KERPLE."""
        attention = SoftmaxAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        # Should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            attention(self.x, rpe=self.rpe)

        # Error message should mention KERPLE incompatibility
        self.assertIn("KERPLE", str(context.exception))
        self.assertIn("kernelized", str(context.exception))

    def test_gradient_flow_through_kerple(self):
        """Test gradients flow through entire FAVOR+ + KERPLE pipeline."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        # Zero gradients
        self.rpe.zero_grad()

        # Forward pass
        output = attention(self.x, rpe=self.rpe)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # RPE parameters should have gradients
        self.assertIsNotNone(self.rpe.rel_pos_bias.grad)
        self.assertTrue(torch.any(self.rpe.rel_pos_bias.grad != 0))


class TestModelCreation(unittest.TestCase):
    """Test model factory with KERPLE variants."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MNISTConfig().to_dict()

    def test_create_performer_favor_most_general(self):
        """Verify model creation: performer_favor_most_general."""
        model = create_model('performer_favor_most_general', self.config)

        # Should create without error
        self.assertIsNotNone(model)

        # Should have the right architecture
        self.assertEqual(model.depth, self.config['depth'])
        self.assertEqual(model.dim, self.config['dim'])

    def test_create_performer_relu_most_general(self):
        """Verify model creation: performer_relu_most_general."""
        model = create_model('performer_relu_most_general', self.config)

        self.assertIsNotNone(model)

    def test_forward_pass(self):
        """Test forward pass through complete model."""
        model = create_model('performer_favor_most_general', self.config)

        # Create sample input (MNIST-like)
        x = torch.randn(2, 1, 28, 28)

        # Forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))  # MNIST has 10 classes

        # Check for NaN/Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestTrainingStability(unittest.TestCase):
    """Test that KERPLE models train stably."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.config = MNISTConfig().to_dict()
        self.model = create_model('performer_favor_most_general', self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def test_single_training_step(self):
        """Verify single training step completes without divergence."""
        # Create dummy batch
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        # Forward pass
        output = self.model(x)
        loss = self.criterion(output, y)

        # Check loss is finite
        self.assertTrue(torch.isfinite(loss))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Check gradients are finite
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Non-finite gradient in {name}"
                )

        # Optimizer step
        self.optimizer.step()

    def test_multiple_training_steps(self):
        """Verify model trains for multiple steps without divergence."""
        losses = []

        for step in range(10):
            # Create dummy batch
            x = torch.randn(4, 1, 28, 28)
            y = torch.randint(0, 10, (4,))

            # Training step
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            # Record loss
            losses.append(loss.item())

            # Check loss is finite
            self.assertTrue(
                math.isfinite(loss.item()),
                f"Loss diverged at step {step}: {loss.item()}"
            )

        # Losses should not explode (some reasonable upper bound)
        for i, loss_val in enumerate(losses):
            self.assertLess(
                loss_val,
                100.0,
                f"Loss too large at step {i}: {loss_val}"
            )


class TestNumericalProperties(unittest.TestCase):
    """Test numerical properties of KERPLE."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.dim = 64
        self.heads = 4

        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)

        self.rpe = KERPLEPositionalEncoding(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        )

    def test_output_magnitude(self):
        """Verify attention output has reasonable magnitude."""
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            dropout=0.0
        )

        output = attention(self.x, rpe=self.rpe)

        # Output should have similar scale to input
        input_std = self.x.std()
        output_std = output.std()

        # Should be within same order of magnitude
        ratio = output_std / input_std
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 10.0)

    def test_fft_numerical_error(self):
        """Measure numerical error of FFT vs naive."""
        n = 32
        d = 8
        c = torch.randn(2 * n - 1)
        x = torch.randn(n, d)

        result_fft = fft_toeplitz_matmul(c, x)
        result_naive = naive_toeplitz_matmul(c, x)

        # Compute relative error
        error = torch.norm(result_fft - result_naive) / torch.norm(result_naive)

        # Should be very small (< 1e-4)
        self.assertLess(error.item(), 1e-4)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFFTToeplitzMultiplication))
    suite.addTests(loader.loadTestsFromTestCase(TestKERPLEPositionalEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestKERPLEIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingStability))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalProperties))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
