"""
Configuration for MNIST dataset experiments.

This configuration matches the hyperparameters from the original notebook
to ensure reproducibility and validation.
"""

MNIST_CONFIG = {
    # Model architecture
    'image_size': 28,         # MNIST images are 28x28
    'in_channels': 1,         # Grayscale images
    'patch_size': 7,          # 4x4 patches -> 7x7 grid = 49 patches
    'num_classes': 10,        # 10 digits (0-9)
    'dim': 32,                # Embedding dimension (same as notebook)
    'depth': 3,               # Number of transformer blocks (same as notebook)
    'heads': 2,               # Number of attention heads (same as notebook)
    'mlp_dim': 64,          # MLP hidden dimension (2x embedding dim)
    'dropout': 0.1,          # Dropout rate

    # Training hyperparameters
    'batch_size': 32,        # Same as notebook
    'learning_rate': 0.001,  # Adam learning rate
    'weight_decay': 0.0,     # No weight decay in original
    'epochs': 10,            # Number of training epochs
    'warmup_epochs': 0,      # No warmup in original

    # Data preprocessing
    'mean': (0.1307,),       # MNIST dataset mean
    'std': (0.3081,),        # MNIST dataset std

    # Data augmentation (none for fair comparison)
    'augmentation': False,

    # Training settings
    'num_workers': 0,        # DataLoader workers (0 to avoid NumPy multiprocessing issues)
    'pin_memory': True,      # Pin memory for GPU
    'seed': 42,              # Random seed for reproducibility

    # Evaluation settings
    'eval_batch_size': 32,   # Batch size for evaluation
    'eval_frequency': 1,     # Evaluate every N epochs

    # Logging and checkpointing
    'log_interval': 100,     # Log every N batches
    'save_checkpoints': True,
    'checkpoint_dir': './checkpoints/mnist',

    # Expected performance (from notebook)
    'expected_accuracy': 0.97,  # ~97-98% after 10 epochs
    'expected_params': 51000,   # ~51K parameters
}
