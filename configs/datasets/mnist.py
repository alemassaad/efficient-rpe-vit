"""
MNIST dataset configuration.

This module defines the configuration for training models on MNIST.
"""

from configs.base import BaseConfig


class MNISTConfig(BaseConfig):
    """Configuration for MNIST dataset."""

    # Dataset-specific parameters
    IMAGE_SIZE = 28
    IN_CHANNELS = 1  # Grayscale
    PATCH_SIZE = 7  # Creates 4x4 = 16 patches
    NUM_CLASSES = 10

    # Model architecture (optimized for MNIST)
    DIM = 32  # Smaller model for simple dataset
    DEPTH = 3
    HEADS = 2
    MLP_DIM = 64
    DROPOUT = 0.1

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0
    EPOCHS = 10
    WARMUP_EPOCHS = 0

    # Data preprocessing
    MEAN = (0.1307,)  # MNIST mean
    STD = (0.3081,)   # MNIST std
    AUGMENTATION = False  # Usually not needed for MNIST

    # Data loading
    NUM_WORKERS = 0  # Often better for small datasets
    PIN_MEMORY = True

    # You can override attention/RPE params here if needed
    # ATTENTION_PARAMS = {
    #     'favor_plus': {
    #         'num_features': 64,  # Override auto-computation
    #         'use_orthogonal': True,
    #     }
    # }


# Create a dictionary for backward compatibility
MNIST_CONFIG = MNISTConfig.to_dict()