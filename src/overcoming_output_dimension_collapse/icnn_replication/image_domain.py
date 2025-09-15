from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.v2 import InterpolationMode, Resize, CenterCrop

"""Definition of the common image domain.

- Channel axis: 1
- Pixel range: [0, 1]
- Image size: arbitrary
- Color space: RGB
"""

# DEFAULT_CHANNEL_AXIS = 1
# DEFAULT_MIN_PIXEL_VALUE = -0.5
# DEFAULT_MAX_PIXEL_VALUE = 0.5
# DEFAULT_PIXEL_RANGE = DEFAULT_MAX_PIXEL_VALUE - DEFAULT_MIN_PIXEL_VALUE


def _bgr2rgb(images: torch.Tensor) -> torch.Tensor:
    """Convert images from BGR to RGB"""
    return images[:, [2, 1, 0], ...]


def _rgb2bgr(images: torch.Tensor) -> torch.Tensor:
    """Convert images from RGB to BGR"""
    return images[:, [2, 1, 0], ...]


def _to_channel_first(images: torch.Tensor) -> torch.Tensor:
    """Convert images from channel last to channel first"""
    return images.permute(0, 3, 1, 2)


def _to_channel_last(images: torch.Tensor) -> torch.Tensor:
    """Convert images from channel first to channel last"""
    return images.permute(0, 2, 3, 1)


def finalize(images: torch.Tensor) -> torch.Tensor:
    """Finalize images as it is ready to be shown."""
    return PILDomainWithExplicitCrop().receive(images)


class ImageDomain(nn.Module, ABC):
    """Image domain which defines the image size, the type of channel axis,
    and the range of pixel values.
    """

    @abstractmethod
    def send(self, images: torch.Tensor) -> torch.Tensor:
        """Send images to the common space from original domain."""
        pass

    @abstractmethod
    def receive(self, images: torch.Tensor) -> torch.Tensor:
        """Receive images from the common space and convert them to the
        original domain."""
        pass


class ComposedDomain(ImageDomain):
    """Composed image domain which is composed of multiple image domains."""

    def __init__(self, domains: list[ImageDomain]) -> None:
        super().__init__()
        self.domains = nn.ModuleList(domains)

    def send(self, images: torch.Tensor) -> torch.Tensor:
        for domain in reversed(self.domains):
            images = domain.send(images)
        return images

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        for domain in self.domains:
            images = domain.receive(images)
        return images


class Zero2OneImageDomain(ImageDomain):
    """Image domain for images in [0, 1]"""

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return images
        # return images * DEFAULT_PIXEL_RANGE + DEFAULT_MIN_PIXEL_VALUE

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return images
        # return (images - DEFAULT_MIN_PIXEL_VALUE) / DEFAULT_PIXEL_RANGE


BareImageDomain = Zero2OneImageDomain


class CenterShiftedDomain(ImageDomain):
    """Image domain for images shifted with centering vector."""

    def __init__(
        self,
        center: np.ndarray,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.center = torch.from_numpy(
            center[np.newaxis, :, np.newaxis, np.newaxis]
        ).to(device=device, dtype=dtype)

    def send(self, images: torch.Tensor) -> torch.Tensor:
        # range of images: [-center, 255-center]
        # target range: [0, 1]
        return (images + self.center) / 255.0

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        # range of images: [0, 1]
        # target range: [-center, 255-center]
        return images * 255.0 - self.center


class BGRDomain(ImageDomain):
    """Image domain for BGR images."""

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return _bgr2rgb(images)

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return _rgb2bgr(images)


class PILDomainWithExplicitCrop(ImageDomain):
    """Image domain for PIL images.

    - Channel axis: 3
    - Pixel range: [0, 255]
    - Image size: arbitrary
    - Color space: RGB
    """

    def send(self, images: torch.Tensor) -> torch.Tensor:
        return _to_channel_first(images) / 255.0  # to [0, 1.0]

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        images = _to_channel_last(images) * 255.0

        # Crop values to [0, 255]
        return torch.clamp(images, 0, 255)


class BdPyVGGDomain(ComposedDomain):
    """Image domain for VGG architecture defined in BdPy.

    - Channel axis: 1
    - Pixel range:
        - red: [-123, 132]
        - green: [-117, 138]
        - blue: [-104, 151]
        # only subtracted mean values ([123, 117, 104]) on ILSVRC are considered
    - Image size: arbitrary
    - Color space: RGB
    """

    def __init__(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(
            [
                CenterShiftedDomain(
                    center=np.array([123.0, 117.0, 104.0]),
                    device=device,
                    dtype=dtype,
                ),
                BGRDomain(),
            ]
        )


class FixedResolutionDomain(ImageDomain):
    """Image domain for images with fixed resolution."""

    def __init__(
        self,
        image_shape: tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.interpolation = interpolation
        self.antialias = antialias

        self.resizer = Resize(
            size=image_shape, interpolation=interpolation, antialias=antialias
        )

    def send(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This domain is not supposed to be used for sending because the"
            "internal image size could not be determined."
        )

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return self.resizer(images)

class CenterCropDomain(ImageDomain):
    """Image domain for images with fixed resolution."""

    def __init__(
        self,
        image_shape: tuple[int, int],
        # interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        # antialias: bool = True,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        # self.interpolation = interpolation
        # self.antialias = antialias

        self.resizer = CenterCrop(
            size=image_shape
        )

    def send(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This domain is not supposed to be used for sending because the"
            "internal image size could not be determined."
        )

    def receive(self, images: torch.Tensor) -> torch.Tensor:
        return self.resizer(images)
