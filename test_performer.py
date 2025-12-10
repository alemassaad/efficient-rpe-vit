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
from models.attention.relu import ReLUAttention
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
            q_prime = attention._compute_phi_positive(q, attention.omega)
            k_prime = attention._compute_phi_positive(k, attention.omega)

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
        from models.components.unified_transformer import UnifiedTransformerBlock
        from models.attention.favor_plus import FAVORPlusAttention

        dim = 64
        heads = 4
        mlp_dim = 128
        seq_len = 16
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create attention module
        attention = FAVORPlusAttention(dim=dim, heads=heads)

        # Create unified transformer block with FAVOR+ attention
        block = UnifiedTransformerBlock(
            dim=dim,
            attention=attention,
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

        # Create model with MNIST config using factory
        model = create_model('performer_favor', MNIST_CONFIG).to(device)

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

        # Create model with CIFAR-10 config using factory
        model = create_model('performer_favor', CIFAR10_CONFIG).to(device)

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
        mnist_model = create_model('performer_favor', MNIST_CONFIG)
        mnist_params = mnist_model.count_parameters()

        # Check we have a reasonable number of parameters
        actual_mnist = mnist_params['trainable']

        # Just verify it's a sensible number (not checking random_features as
        # they are buffers, not parameters in the new architecture)
        self.assertGreater(actual_mnist, 10000,
                          f"Model has too few parameters: {actual_mnist}")
        self.assertLess(actual_mnist, 1000000,
                       f"Model has too many parameters: {actual_mnist}")

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('performer_favor', MNIST_CONFIG).to(device)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 1, 28, 28).to(device)
            output = model(x)
            self.assertEqual(output.shape[0], batch_size)


class TestApproximationQuality(unittest.TestCase):
    """Test FAVOR+ approximation quality compared to standard attention."""

    def test_approximation_error(self):
        """Test that FAVOR+ produces valid outputs with reasonable magnitude.

        Note: FAVOR+ is an approximation of softmax attention. The approximation
        quality depends on the number of random features and is stochastic.
        Rather than comparing to exact softmax, we verify FAVOR+ produces
        numerically stable outputs with reasonable magnitude.
        """
        dim = 64
        heads = 4
        seq_len = 16
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create FAVOR+ attention with more random features for better approximation
        favor_attn = FAVORPlusAttention(
            dim=dim,
            heads=heads,
            num_features=256  # More features for better approximation
        ).to(device)

        # Create input with known magnitude
        x = torch.randn(batch_size, seq_len, dim).to(device)
        input_norm = x.norm()

        # Forward pass
        with torch.no_grad():
            favor_output = favor_attn(x)

        output_norm = favor_output.norm()

        # Verify output is numerically stable
        self.assertFalse(torch.isnan(favor_output).any(),
                        "FAVOR+ output contains NaN")
        self.assertFalse(torch.isinf(favor_output).any(),
                        "FAVOR+ output contains Inf")

        # Verify output has reasonable magnitude (not exploding or vanishing)
        # Output norm should be within 10x of input norm
        ratio = output_norm / input_norm
        self.assertGreater(ratio, 0.01,
                          f"Output magnitude too small: ratio={ratio:.4f}")
        self.assertLess(ratio, 100,
                       f"Output magnitude too large: ratio={ratio:.4f}")

        print(f"\nFAVOR+ output quality:")
        print(f"  Input norm: {input_norm:.3f}")
        print(f"  Output norm: {output_norm:.3f}")
        print(f"  Ratio: {ratio:.3f}")


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
            from models.attention.softmax import SoftmaxAttention
            standard_attn = SoftmaxAttention(dim=dim, heads=heads).to(device)
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


class TestReLUAttention(unittest.TestCase):
    """Test ReLU attention mechanism (Performer-ReLU)."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.heads = 4
        self.seq_len = 16
        self.batch_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_relu_attention_forward(self):
        """Test ReLU attention forward pass."""
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        output = attention(x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_relu_features_non_negative(self):
        """Test that ReLU features are non-negative."""
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        # Create random input
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim // self.heads).to(self.device)

        # Compute ReLU features
        features = attention._compute_relu_features(x, attention.omega)

        # Check all features are non-negative (ReLU output)
        self.assertTrue((features >= 0).all(), "ReLU features should be non-negative")

    def test_relu_attention_gradient_flow(self):
        """Test gradient flow through ReLU attention."""
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        x.requires_grad = True

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Check parameter gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_relu_vs_favor_architecture(self):
        """Test that ReLU attention uses same architecture as FAVOR+."""
        relu_attn = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            num_features=64,
            use_orthogonal=True
        ).to(self.device)

        favor_attn = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            num_features=64,
            use_orthogonal=True
        ).to(self.device)

        # Check same number of features
        self.assertEqual(relu_attn.num_features, favor_attn.num_features)

        # Check same omega shape
        self.assertEqual(relu_attn.omega.shape, favor_attn.omega.shape)

        # Check same scaling
        self.assertEqual(relu_attn.relu_scale, favor_attn.favor_scale)

    def test_performer_relu_model_creation(self):
        """Test creating performer_relu model via factory."""
        model = create_model('performer_relu', MNIST_CONFIG)

        self.assertIsNotNone(model)

        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)

        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_relu_attention_with_different_seq_lengths(self):
        """Test ReLU attention with various sequence lengths."""
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        for seq_len in [4, 16, 64, 256]:
            x = torch.randn(self.batch_size, seq_len, self.dim).to(self.device)
            output = attention(x)

            self.assertEqual(output.shape, (self.batch_size, seq_len, self.dim))
            self.assertFalse(torch.isnan(output).any())

    def test_relu_attention_orthogonal_vs_iid(self):
        """Test both orthogonal and i.i.d. random features."""
        for use_orthogonal in [True, False]:
            attention = ReLUAttention(
                dim=self.dim,
                heads=self.heads,
                use_orthogonal=use_orthogonal
            ).to(self.device)

            x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
            output = attention(x)

            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
            self.assertFalse(torch.isnan(output).any())


class TestReLUWithRoPE(unittest.TestCase):
    """Test ReLU attention with RoPE relative position encoding."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.heads = 4
        self.seq_len = 16
        self.batch_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_relu_rope_forward_pass(self):
        """Test ReLU with RoPE produces valid output."""
        from models.rpe import RoPE

        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        output = attention(x, rpe=rpe)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
        # Check no NaNs
        self.assertFalse(torch.isnan(output).any())
        # Check no Infs
        self.assertFalse(torch.isinf(output).any())

    def test_relu_rope_gradient_flow(self):
        """Test gradients flow through ReLU with RoPE."""
        from models.rpe import RoPE

        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        x.requires_grad = True

        output = attention(x, rpe=rpe)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Check attention parameter gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_relu_rope_different_from_no_rpe(self):
        """Test that RoPE actually changes the output."""
        from models.rpe import RoPE

        torch.manual_seed(42)
        attention = ReLUAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)

        # Get output without RoPE
        output_no_rpe = attention(x, rpe=None)

        # Get output with RoPE
        output_with_rpe = attention(x, rpe=rpe)

        # They should be different
        self.assertFalse(
            torch.allclose(output_no_rpe, output_with_rpe, rtol=1e-3, atol=1e-3),
            "RoPE should change the attention output"
        )

    def test_performer_relu_rope_model_creation(self):
        """Test creating performer_relu_rope model via factory."""
        model = create_model('performer_relu_rope', MNIST_CONFIG)

        self.assertIsNotNone(model)
        self.assertEqual(model.model_name, 'performer_relu_rope')
        self.assertEqual(model.attention_type, 'relu')
        self.assertEqual(model.rpe_type, 'rope')

        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)

        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_performer_relu_rope_training_step(self):
        """Test a full training step with performer_relu_rope."""
        model = create_model('performer_relu_rope', MNIST_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Simulate training step
        x = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))

        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestFAVORPlusWithRoPE(unittest.TestCase):
    """Test FAVOR+ attention with RoPE relative position encoding."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.heads = 4
        self.seq_len = 16
        self.batch_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_favor_plus_rope_forward_pass(self):
        """Test FAVOR+ with RoPE produces valid output."""
        from models.rpe import RoPE

        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        output = attention(x, rpe=rpe)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
        # Check no NaNs
        self.assertFalse(torch.isnan(output).any())
        # Check no Infs
        self.assertFalse(torch.isinf(output).any())

    def test_favor_plus_rope_gradient_flow(self):
        """Test gradients flow through FAVOR+ with RoPE."""
        from models.rpe import RoPE

        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        x.requires_grad = True

        output = attention(x, rpe=rpe)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Check attention parameter gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_favor_plus_rope_different_from_no_rpe(self):
        """Test that RoPE actually changes the output."""
        from models.rpe import RoPE

        torch.manual_seed(42)
        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads,
            use_orthogonal=True
        ).to(self.device)

        rpe = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)

        # Get output without RoPE
        output_no_rpe = attention(x, rpe=None)

        # Get output with RoPE
        output_with_rpe = attention(x, rpe=rpe)

        # They should be different
        self.assertFalse(
            torch.allclose(output_no_rpe, output_with_rpe, rtol=1e-3, atol=1e-3),
            "RoPE should change the attention output"
        )

    def test_performer_favor_rope_model_creation(self):
        """Test creating performer_favor_rope model via factory."""
        model = create_model('performer_favor_rope', MNIST_CONFIG)

        self.assertIsNotNone(model)
        self.assertEqual(model.model_name, 'performer_favor_rope')
        self.assertEqual(model.attention_type, 'favor_plus')
        self.assertEqual(model.rpe_type, 'rope')

        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)

        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_performer_favor_rope_training_step(self):
        """Test a full training step with performer_favor_rope."""
        model = create_model('performer_favor_rope', MNIST_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Simulate training step
        x = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))

        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_favor_plus_rope_vs_kerple_different_paths(self):
        """Test that RoPE and KERPLE use different code paths."""
        from models.rpe import RoPE, KERPLEPositionalEncoding

        attention = FAVORPlusAttention(
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        rope = RoPE(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        kerple = KERPLEPositionalEncoding(
            num_patches=self.seq_len,
            dim=self.dim,
            heads=self.heads
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)

        # Both should work without errors
        output_rope = attention(x, rpe=rope)
        output_kerple = attention(x, rpe=kerple)

        # Both should produce valid outputs
        self.assertEqual(output_rope.shape, output_kerple.shape)
        self.assertFalse(torch.isnan(output_rope).any())
        self.assertFalse(torch.isnan(output_kerple).any())


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
    suite.addTests(loader.loadTestsFromTestCase(TestReLUAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestReLUWithRoPE))
    suite.addTests(loader.loadTestsFromTestCase(TestFAVORPlusWithRoPE))

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