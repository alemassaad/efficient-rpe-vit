"""
CIFAR-10 dataset configuration.

This module defines the configuration for training models on CIFAR-10.
"""

from configs.base import BaseConfig


class CIFAR10Config(BaseConfig):
    """Configuration for CIFAR-10 dataset."""

    # Dataset-specific parameters
    IMAGE_SIZE = 32
    IN_CHANNELS = 3  # RGB
    PATCH_SIZE = 8  # Creates 4x4 = 16 patches
    NUM_CLASSES = 10

    # Model architecture (can be larger than MNIST)
    DIM = 32  # Could increase for better performance
    DEPTH = 3
    HEADS = 2
    MLP_DIM = 64
    DROPOUT = 0.1

    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01  # Some regularization helps
    EPOCHS = 20
    WARMUP_EPOCHS = 2

    # Data preprocessing
    MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR-10 channel means
    STD = (0.2470, 0.2435, 0.2616)   # CIFAR-10 channel stds
    AUGMENTATION = False  # Could enable for better generalization

    # Data loading
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # You can override attention/RPE params here if needed
    # RPE_PARAMS = {
    #     'rope': {
    #         'theta': 5000.0  # Different base frequency for CIFAR
    #     }
    # }


# Create a dictionary for backward compatibility
CIFAR10_CONFIG = CIFAR10Config.to_dict()