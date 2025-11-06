"""
CIFAR-10 configuration (backward compatibility).

This file maintains backward compatibility with old code.
New code should use: from configs.datasets.cifar10 import CIFAR10_CONFIG
"""

from configs.datasets.cifar10 import CIFAR10_CONFIG, CIFAR10Config

# Re-export for backward compatibility
__all__ = ['CIFAR10_CONFIG', 'CIFAR10Config']
