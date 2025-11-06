"""
Configuration system for Vision Transformer experiments.

This package provides structured configuration for different datasets and model variants.
"""

from .base import BaseConfig, get_attention_config, get_rpe_config
from .datasets.mnist import MNISTConfig, MNIST_CONFIG
from .datasets.cifar10 import CIFAR10Config, CIFAR10_CONFIG

__all__ = [
    'BaseConfig',
    'MNISTConfig',
    'CIFAR10Config',
    'MNIST_CONFIG',
    'CIFAR10_CONFIG',
    'get_attention_config',
    'get_rpe_config',
]