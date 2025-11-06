"""
Most General RPE implementation stub.

Based on "Improve Transformer Models with Better Relative Position Embeddings"
by Luo et al., 2021.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseRPE


class MostGeneralRPE(BaseRPE):
    """
    Most General Relative Position Encoding (stub implementation).

    This is a placeholder for the Most General RPE from Luo et al., 2021.
    The actual implementation will be added after researching the paper's
    specific requirements and mathematical formulation.

    Args:
        num_patches: Number of patches in the sequence
        dim: Model dimension
        heads: Number of attention heads
        **kwargs: Additional parameters for future implementation
    """

    def __init__(
        self,
        num_patches: int,
        dim: int,
        heads: int,
        **kwargs
    ):
        super().__init__(num_patches, dim, heads)

        # Store additional parameters for future use
        self.additional_params = kwargs

        # Placeholder for learnable parameters
        # The actual parameters will depend on the paper's formulation
        self.rpe_table = nn.Parameter(
            torch.zeros(1, heads, num_patches, num_patches),
            requires_grad=False  # Disabled until implementation
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Most General RPE (not implemented).

        Args:
            x: Input tensor
            attention_scores: Optional attention scores

        Returns:
            Output tensor (unchanged in stub)

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "Most General RPE is not yet implemented. "
            "This is a placeholder for future development based on "
            "Luo et al., 2021. The exact integration point and "
            "mathematical formulation need to be researched."
        )

    def extra_repr(self) -> str:
        """String representation."""
        return super().extra_repr() + ', status=NOT_IMPLEMENTED'