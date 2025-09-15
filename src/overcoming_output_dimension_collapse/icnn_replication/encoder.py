from __future__ import annotations

import torch
import torch.nn as nn
from bdpy.dl.torch import FeatureExtractor

from . import image_domain


class Encoder(nn.Module):
    """Encoder network module.

    Parameters
    ----------
    feature_network : nn.Module
        Feature network. This network should have a method `forward` that takes
        an image tensor and propagates it through the network.
    layer_names : list[str]
        Layer names to extract features from.
    domain : image_domain.ImageDomain
        Image domain to receive images.
    device : torch.device
        Device to use.
    """

    def __init__(
        self,
        feature_network: nn.Module,
        layer_names: list[str],
        domain: image_domain.ImageDomain,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            encoder=feature_network, layers=layer_names, detach=False, device=device
        )
        self.domain = domain
        self.feature_network = self.feature_extractor._encoder

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the encoder network.

        Parameters
        ----------
        images : torch.Tensor
            Images.

        Returns
        -------
        dict[str, torch.Tensor]
            Features indexed by the layer names.
        """
        images = self.domain.receive(images)
        return self.feature_extractor(images)
