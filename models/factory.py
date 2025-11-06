"""
Factory module for creating Vision Transformer models.

This module provides a unified interface for creating any of the 9+ model
variants by combining different attention mechanisms and RPE types.
"""

import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from functools import partial

from .core.base_vit import BaseViT
from .attention import ATTENTION_REGISTRY
from .rpe import RPE_REGISTRY


# Model name mappings to (attention_type, rpe_type)
MODEL_VARIANTS = {
    # Baseline models
    'baseline': ('softmax', None),
    'baseline_most_general': ('softmax', 'most_general'),
    'baseline_circulant': ('softmax', 'circulant_string'),
    'baseline_rope': ('softmax', 'rope'),

    # Performer FAVOR+ models
    'performer_favor': ('favor_plus', None),
    'performer_favor_most_general': ('favor_plus', 'most_general'),
    'performer_favor_circulant': ('favor_plus', 'circulant_string'),
    'performer_favor_rope': ('favor_plus', 'rope'),

    # Performer ReLU models
    'performer_relu': ('relu', None),
    'performer_relu_most_general': ('relu', 'most_general'),
    'performer_relu_circulant': ('relu', 'circulant_string'),
    'performer_relu_rope': ('relu', 'rope'),

    # Aliases for convenience
    'performer': ('favor_plus', None),  # Default performer is FAVOR+
    'vit': ('softmax', None),  # Standard ViT
}


def create_attention_builder(
    attention_type: str,
    attention_config: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a builder function for attention modules.

    Args:
        attention_type: Type of attention mechanism
        attention_config: Additional configuration for attention

    Returns:
        Function that creates attention modules with the specified config
    """
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type: {attention_type}. "
            f"Available types: {list(ATTENTION_REGISTRY.keys())}"
        )

    attention_class = ATTENTION_REGISTRY[attention_type]
    config = attention_config or {}

    def builder(dim: int, heads: int, dropout: float = 0.0) -> nn.Module:
        """Build attention module with specified dimensions."""
        return attention_class(
            dim=dim,
            heads=heads,
            dropout=dropout,
            **config
        )

    return builder


def create_rpe_builder(
    rpe_type: Optional[str],
    rpe_config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """
    Create a builder function for RPE modules.

    Args:
        rpe_type: Type of RPE mechanism (or None)
        rpe_config: Additional configuration for RPE

    Returns:
        Function that creates RPE modules, or None if no RPE
    """
    if rpe_type is None:
        return None

    if rpe_type not in RPE_REGISTRY:
        raise ValueError(
            f"Unknown RPE type: {rpe_type}. "
            f"Available types: {list(RPE_REGISTRY.keys())}"
        )

    rpe_class = RPE_REGISTRY[rpe_type]
    config = rpe_config or {}

    def builder(num_patches: int, dim: int, heads: int) -> nn.Module:
        """Build RPE module with specified dimensions."""
        return rpe_class(
            num_patches=num_patches,
            dim=dim,
            heads=heads,
            **config
        )

    return builder


def create_model(
    model_name: str,
    dataset_config: Dict[str, Any],
    attention_config: Optional[Dict[str, Any]] = None,
    rpe_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseViT:
    """
    Create a Vision Transformer model with specified configuration.

    This is the main entry point for creating any model variant.

    Args:
        model_name: Name of the model variant (e.g., 'baseline', 'performer_favor_rope')
        dataset_config: Configuration dictionary with dataset-specific parameters
                       (image_size, num_classes, etc.)
        attention_config: Optional attention-specific configuration
        rpe_config: Optional RPE-specific configuration
        **kwargs: Override any configuration parameters

    Returns:
        Configured Vision Transformer model

    Examples:
        >>> # Create baseline model for MNIST
        >>> from configs.mnist_config import MNIST_CONFIG
        >>> model = create_model('baseline', MNIST_CONFIG)

        >>> # Create Performer with RoPE for CIFAR-10
        >>> from configs.cifar10_config import CIFAR10_CONFIG
        >>> model = create_model('performer_favor_rope', CIFAR10_CONFIG)

        >>> # Create model with custom parameters
        >>> model = create_model(
        ...     'performer_favor',
        ...     MNIST_CONFIG,
        ...     attention_config={'num_features': 256, 'use_orthogonal': True},
        ...     dropout=0.2
        ... )
    """
    # Parse model name to get attention and RPE types
    if model_name in MODEL_VARIANTS:
        attention_type, rpe_type = MODEL_VARIANTS[model_name]
    else:
        # Try to parse custom model name (e.g., "custom_attention_rpe")
        parts = model_name.split('_')
        if len(parts) < 1:
            raise ValueError(f"Invalid model name: {model_name}")

        # Assume first part is attention type
        attention_type = parts[0]
        # Rest could be RPE type
        rpe_type = '_'.join(parts[1:]) if len(parts) > 1 else None

        # Validate
        if attention_type not in ATTENTION_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(MODEL_VARIANTS.keys())}"
            )

    # Merge configurations
    config = dataset_config.copy()
    config.update(kwargs)

    # Extract model-specific configs if they exist in dataset_config
    if 'attention_params' in config:
        default_attention_config = config['attention_params'].get(attention_type, {})
        if attention_config:
            default_attention_config.update(attention_config)
        attention_config = default_attention_config
        del config['attention_params']  # Remove from base config

    if 'rpe_params' in config and rpe_type:
        default_rpe_config = config['rpe_params'].get(rpe_type, {})
        if rpe_config:
            default_rpe_config.update(rpe_config)
        rpe_config = default_rpe_config
        del config['rpe_params']  # Remove from base config

    # Create builder functions
    attention_builder = create_attention_builder(attention_type, attention_config)
    rpe_builder = create_rpe_builder(rpe_type, rpe_config)

    # Extract only the parameters that BaseViT expects
    vit_params = {
        'image_size': config['image_size'],
        'in_channels': config['in_channels'],
        'patch_size': config['patch_size'],
        'num_classes': config['num_classes'],
        'dim': config['dim'],
        'depth': config['depth'],
        'heads': config['heads'],
        'mlp_dim': config['mlp_dim'],
        'dropout': config.get('dropout', 0.1),
        'attention_builder': attention_builder,
        'rpe_builder': rpe_builder,
    }

    # Create the model
    model = BaseViT(**vit_params)

    # Add metadata for tracking
    model.model_name = model_name
    model.attention_type = attention_type
    model.rpe_type = rpe_type

    return model


def list_available_models() -> list:
    """
    List all available pre-configured model variants.

    Returns:
        List of model names
    """
    return list(MODEL_VARIANTS.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model variant.

    Args:
        model_name: Name of the model variant

    Returns:
        Dictionary with model information
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model: {model_name}")

    attention_type, rpe_type = MODEL_VARIANTS[model_name]

    return {
        'name': model_name,
        'attention_type': attention_type,
        'rpe_type': rpe_type,
        'attention_complexity': 'O(N²)' if attention_type == 'softmax' else 'O(N)',
        'has_rpe': rpe_type is not None,
    }