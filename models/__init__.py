"""
Vision Transformer models with various attention mechanisms and RPE types.

This package provides a unified interface for creating and using different
Vision Transformer variants, including:
- Standard ViT with softmax attention
- Performer with FAVOR+ linear attention
- Support for various Relative Positional Encodings (RPE)

Main entry point: create_model()
"""

from .factory import (
    create_model,
    list_available_models,
    get_model_info,
    MODEL_VARIANTS
)

from .core.base_vit import BaseViT
from .attention import (
    SoftmaxAttention,
    FAVORPlusAttention,
    ReLUAttention,
    ATTENTION_REGISTRY
)
from .rpe import (
    KERPLEPositionalEncoding,
    CirculantStringRPE,
    RoPE,
    RPE_REGISTRY
)

__all__ = [
    # Main factory function
    'create_model',
    'list_available_models',
    'get_model_info',
    'MODEL_VARIANTS',

    # Base classes
    'BaseViT',

    # Attention mechanisms
    'SoftmaxAttention',
    'FAVORPlusAttention',
    'ReLUAttention',
    'ATTENTION_REGISTRY',

    # RPE mechanisms
    'KERPLEPositionalEncoding',
    'CirculantStringRPE',
    'RoPE',
    'RPE_REGISTRY',
]

# For backward compatibility (will remove later)
# These allow old code to still work temporarily
def create_baseline_vit(config: dict):
    """Deprecated: Use create_model('baseline', config) instead."""
    import warnings
    warnings.warn(
        "create_baseline_vit is deprecated. Use create_model('baseline', config) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_model('baseline', config)


def create_performer_vit(config: dict):
    """Deprecated: Use create_model('performer_favor', config) instead."""
    import warnings
    warnings.warn(
        "create_performer_vit is deprecated. Use create_model('performer_favor', config) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Extract FAVOR+ specific params from config
    attention_config = {
        'num_features': config.get('num_features'),
        'use_orthogonal': config.get('use_orthogonal', True),
        'feature_redraw_interval': config.get('feature_redraw_interval')
    }
    # Remove them from base config
    base_config = {k: v for k, v in config.items()
                   if k not in ['num_features', 'use_orthogonal', 'feature_redraw_interval']}

    return create_model('performer_favor', base_config, attention_config=attention_config)