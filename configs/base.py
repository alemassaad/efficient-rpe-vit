"""
Base configuration shared by all datasets and models.

This module defines the common structure and default values for all configurations.
"""

from typing import Dict, Any, Optional


class BaseConfig:
    """Base configuration class with common parameters."""

    # Model architecture (common to all variants)
    IMAGE_SIZE: int = None  # Must be set by dataset config
    IN_CHANNELS: int = None  # Must be set by dataset config
    PATCH_SIZE: int = None  # Must be set by dataset config
    NUM_CLASSES: int = None  # Must be set by dataset config

    DIM: int = 64  # Model dimension
    DEPTH: int = 3  # Number of transformer blocks
    HEADS: int = 4  # Number of attention heads
    MLP_DIM: int = 256  # MLP hidden dimension
    DROPOUT: float = 0.1  # Dropout rate

    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0
    EPOCHS: int = 10
    WARMUP_EPOCHS: int = 0

    # Data preprocessing
    MEAN: tuple = None  # Must be set by dataset config
    STD: tuple = None  # Must be set by dataset config
    AUGMENTATION: bool = False

    # Data loading
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    # Random seed
    SEED: int = 42

    # Attention-specific parameters
    ATTENTION_PARAMS: Dict[str, Dict[str, Any]] = {
        'softmax': {},  # No special params for standard attention
        'favor_plus': {
            'num_features': None,  # Auto-compute as d*log(d)
            'use_orthogonal': True,
            'feature_redraw_interval': None
        },
        'relu': {}  # Placeholder for future params
    }

    # RPE-specific parameters
    RPE_PARAMS: Dict[str, Dict[str, Any]] = {
        'most_general': {},  # To be defined
        'circulant_string': {},  # To be defined
        'rope': {
            'theta': 10000.0  # Base frequency for rotary encoding
        }
    }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                if value is not None:
                    config[key.lower()] = value
        return config

    @classmethod
    def update(cls, **kwargs) -> Dict[str, Any]:
        """Create updated configuration dictionary."""
        config = cls.to_dict()
        config.update(kwargs)
        return config


def get_attention_config(attention_type: str, base_config: BaseConfig) -> Dict[str, Any]:
    """
    Get attention-specific configuration.

    Args:
        attention_type: Type of attention mechanism
        base_config: Base configuration class

    Returns:
        Dictionary with attention-specific parameters
    """
    if hasattr(base_config, 'ATTENTION_PARAMS'):
        return base_config.ATTENTION_PARAMS.get(attention_type, {})
    return {}


def get_rpe_config(rpe_type: str, base_config: BaseConfig) -> Dict[str, Any]:
    """
    Get RPE-specific configuration.

    Args:
        rpe_type: Type of RPE mechanism
        base_config: Base configuration class

    Returns:
        Dictionary with RPE-specific parameters
    """
    if hasattr(base_config, 'RPE_PARAMS'):
        return base_config.RPE_PARAMS.get(rpe_type, {})
    return {}