"""
Attention mechanisms for Vision Transformer.

This module provides various attention implementations:
- SoftmaxAttention: Standard O(N²) attention
- FAVORPlusAttention: Linear O(N) attention from Performer
- ReLUAttention: Placeholder for ReLU-based linear attention
"""

from .base import BaseAttention
from .softmax import SoftmaxAttention
from .favor_plus import FAVORPlusAttention
from .relu import ReLUAttention

# Registry of available attention mechanisms
ATTENTION_REGISTRY = {
    'softmax': SoftmaxAttention,
    'baseline': SoftmaxAttention,  # Alias for backward compatibility
    'favor_plus': FAVORPlusAttention,
    'favor+': FAVORPlusAttention,  # Alias
    'performer': FAVORPlusAttention,  # Alias
    'relu': ReLUAttention,
}

__all__ = [
    'BaseAttention',
    'SoftmaxAttention',
    'FAVORPlusAttention',
    'ReLUAttention',
    'ATTENTION_REGISTRY',
]