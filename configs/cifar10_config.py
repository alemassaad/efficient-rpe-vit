"""
Configuration for CIFAR-10 dataset experiments.

CIFAR-10 is more complex than MNIST, requiring:
- Larger model capacity (more parameters)
- RGB channels (3 vs 1)
- More training epochs
- Different normalization constants
"""

CIFAR10_CONFIG = {
    # Model architecture
    'image_size': 32,         # CIFAR-10 images are 32x32
    'in_channels': 3,         # RGB images
    'patch_size': 8,          # 4x4 patches -> 8x8 grid = 64 patches
    'num_classes': 10,        # 10 classes (airplane, automobile, bird, etc.)
    'dim': 32,               # Larger embedding dimension for complexity
    'depth': 3,               # More transformer blocks for harder task
    'heads': 2,               # More attention heads
    'mlp_dim': 64,          # MLP hidden dimension (2x embedding dim)
    'dropout': 0.1,          # Dropout rate (can increase if overfitting)

    # Training hyperparameters
    'batch_size': 64,        # Larger batch size if GPU memory allows
    'learning_rate': 0.001,  # Adam learning rate (may need tuning)
    'weight_decay': 0.01,    # Some weight decay for regularization
    'epochs': 20,            # More epochs for harder dataset
    'warmup_epochs': 2,      # Warmup for stable training

    # Data preprocessing (computed from CIFAR-10 training set)
    'mean': (0.4914, 0.4822, 0.4465),  # Per-channel mean
    'std': (0.2470, 0.2435, 0.2616),   # Per-channel std

    # Data augmentation (disabled for fair baseline comparison)
    'augmentation': False,    # Can enable later for better accuracy

    # Training settings
    'num_workers': 0,        # DataLoader workers (0 to avoid NumPy multiprocessing issues)
    'pin_memory': True,      # Pin memory for GPU
    'seed': 42,              # Random seed for reproducibility

    # Evaluation settings
    'eval_batch_size': 64,   # Batch size for evaluation
    'eval_frequency': 1,     # Evaluate every N epochs

    # Logging and checkpointing
    'log_interval': 50,      # Log every N batches
    'save_checkpoints': True,
    'checkpoint_dir': './checkpoints/cifar10',

    # Expected performance (baseline ViT without augmentation)
    'expected_accuracy': 0.70,  # ~65-75% after 20 epochs (no augmentation)
    'expected_params': 200000,  # ~200K parameters

    # Optional: Patch size variations to experiment with
    'alternative_patch_sizes': {
        '8x8': 8,  # 32/8 = 4x4 grid = 16 patches (faster, less accurate)
        '2x2': 2,  # 32/2 = 16x16 grid = 256 patches (slower, more accurate)
    }
}
