from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from . import deep_image_prior, image_domain


@torch.no_grad()
def _reset_weight(m: nn.Module) -> None:
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class Generator(nn.Module, ABC):
    """Generator module.

    Note
    ----
    This module must have the following methods:

    reset_states()
        Reset the state of the generator. This method is called before generating
        images for each stimulus.
    forward()
        Generate images. This method is called for each iteration of the optimization
        process. The forward method has no arguments and the generated images must be
        in the range [0, 1].
    """

    @abstractmethod
    def reset_states(self) -> None:
        """Reset the state of the generator."""
        pass


class DeepImagePriorGenerator(Generator):
    def __init__(
        self,
        image_shape: tuple[int, int],
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert batch_size == 1, "batch_size must be 1 for DeepImagePriorGenerator"

        self._image_shape = image_shape
        self._batch_size = batch_size
        self.unet = deep_image_prior.get_net(
            input_depth=3,
            NET_TYPE="UNet",
            pad="reflection",
            upsample_mode="bilinear",
        )
        self.unet.to(device)
        self.latent_image = nn.Parameter(
            torch.empty(batch_size, 3, *image_shape, **factory_kwargs)
        )
        self.domain = image_domain.Zero2OneImageDomain()
        self.reset_states()

    def reset_states(self) -> None:
        nn.init.uniform_(self.latent_image, 0.0, 1.0)
        # Re-initialize the weights of the UNet
        self.unet.apply(_reset_weight)

    def forward(self) -> torch.Tensor:
        image = self.unet(self.latent_image)
        return self.domain.send(image)
