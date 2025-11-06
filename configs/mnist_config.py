"""
MNIST configuration (backward compatibility).

This file maintains backward compatibility with old code.
New code should use: from configs.datasets.mnist import MNIST_CONFIG
"""

from configs.datasets.mnist import MNIST_CONFIG, MNISTConfig

# Re-export for backward compatibility
__all__ = ['MNIST_CONFIG', 'MNISTConfig']
