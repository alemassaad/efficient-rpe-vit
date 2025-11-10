"""
Relative Positional Encoding mechanisms for Vision Transformer.

This module provides various RPE implementations:
- KERPLEPositionalEncoding: From Luo et al., 2021 (FULLY IMPLEMENTED)
  Also known as "Most General RPE" - uses FFT for O(n log n) complexity
- CirculantStringRPE: From Schenck et al., 2025 (stub)
- RoPE: Rotary Position Embedding from Su et al., 2024 (stub)
"""

from .base import BaseRPE
from .kerple import KERPLEPositionalEncoding
from .circulant_string import CirculantStringRPE
from .rope import RoPE

# Registry of available RPE mechanisms
RPE_REGISTRY = {
    'most_general': KERPLEPositionalEncoding,  # KERPLE is the "Most General RPE" from paper
    'kerple': KERPLEPositionalEncoding,  # Direct name
    'circulant_string': CirculantStringRPE,
    'circulant': CirculantStringRPE,  # Alias
    'rope': RoPE,
    'rotary': RoPE,  # Alias
}

__all__ = [
    'BaseRPE',
    'KERPLEPositionalEncoding',
    'CirculantStringRPE',
    'RoPE',
    'RPE_REGISTRY',
]