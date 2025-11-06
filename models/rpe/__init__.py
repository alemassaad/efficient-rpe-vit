"""
Relative Positional Encoding mechanisms for Vision Transformer.

This module provides various RPE implementations:
- MostGeneralRPE: From Luo et al., 2021 (stub)
- CirculantStringRPE: From Schenck et al., 2025 (stub)
- RoPE: Rotary Position Embedding from Su et al., 2024 (stub)
"""

from .base import BaseRPE
from .most_general import MostGeneralRPE
from .circulant_string import CirculantStringRPE
from .rope import RoPE

# Registry of available RPE mechanisms
RPE_REGISTRY = {
    'most_general': MostGeneralRPE,
    'circulant_string': CirculantStringRPE,
    'circulant': CirculantStringRPE,  # Alias
    'rope': RoPE,
    'rotary': RoPE,  # Alias
}

__all__ = [
    'BaseRPE',
    'MostGeneralRPE',
    'CirculantStringRPE',
    'RoPE',
    'RPE_REGISTRY',
]