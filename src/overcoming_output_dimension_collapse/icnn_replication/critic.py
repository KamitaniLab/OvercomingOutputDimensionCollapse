from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import wandb


class Critic(nn.Module, ABC):
    """Critic module.

    Note
    ----
    This module must have the following methods:

    criterion(feature, target_feature, layer_name)
        Loss function per layer. This method is called for each layer.
    """

    def __init__(self, *, with_wandb: bool = False) -> None:
        super().__init__()
        self.with_wandb = with_wandb

    def enable_wandb(self) -> None:
        self.with_wandb = True

    @abstractmethod
    def criterion(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """Loss function per layer.

        Parameters
        ----------
        feature : torch.Tensor
            Generated feature.
        target_feature : torch.Tensor
            Target feature.
        layer_name : str
            Layer name.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        pass

    def forward(
        self,
        features: dict[str, torch.Tensor],
        target_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through the critic network.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Generated features indexed by the layer names.
        target_features : dict[str, torch.Tensor]
            Target features indexed by the layer names.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = 0.0
        counts = 0
        for layer_name, feature in features.items():
            target_feature = target_features[layer_name]
            layer_wise_loss = self.criterion(
                feature, target_feature, layer_name=layer_name
            )
            if self.with_wandb:
                wandb.log(
                    {f"critic/{layer_name}": layer_wise_loss.mean().item()},
                    commit=False,
                )
            loss += layer_wise_loss
            counts += 1
        return loss / counts


class MSEperTargetNorm(Critic):
    """MSE devided by the target norm."""

    def criterion(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        target_norm = (
            (target_feature**2).sum(dim=tuple(range(1, target_feature.ndim))).sqrt()
        )
        return ((feature - target_feature) ** 2).sum(
            dim=tuple(range(1, target_feature.ndim))
        ) / target_norm


class TargetNormalizedMSE(Critic):
    """MSE over vectors normalized by the target norm."""

    def criterion(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        target_norm = (
            (target_feature**2)
            .sum(dim=tuple(range(1, target_feature.ndim)), keepdim=True)
            .sqrt()
        )
        f = feature / target_norm
        tf = target_feature / target_norm
        return (f - tf).pow(2).sum(dim=tuple(range(1, target_feature.ndim)))


class DistsLoss(Critic):
    def criterion(
        self,
        feature: torch.Tensor,
        target_feature: torch.Tensor,
        layer_name: str,
        eps: float = 1e-6,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> torch.Tensor:
        if "classifier" in layer_name:
            # feature.shape = (batch_size, feature_dim)
            # target_feature.shape = (batch_size, feature_dim)
            feature_mean = feature.mean(dim=1, keepdim=True)  # (batch_size, 1)
            target_feature_mean = target_feature.mean(dim=1, keepdim=True)  # (batch_size, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=1, keepdim=True)  # (batch_size, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(
                dim=1, keepdim=True
            )  # (batch_size, 1)
            cov = (
                (feature - feature_mean) * (target_feature - target_feature_mean)
            ).mean(dim=1, keepdim=True)  # (batch_size, 1)
        else:
            # feature.shape = (batch_size, channel, height, width)
            # target_feature.shape = (batch_size, channel, height, width)
            feature_mean = feature.mean(dim=[2, 3], keepdim=True)  # (batch_size, channel, 1, 1)
            target_feature_mean = target_feature.mean(dim=[2, 3], keepdim=True)  # (batch_size, channel, 1, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=[2, 3], keepdim=True)  # (batch_size, channel, 1, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(
                dim=[2, 3], keepdim=True
            )  # (batch_size, channel, 1, 1)
            cov = (
                (feature - feature_mean) * (target_feature - target_feature_mean)
            ).mean(dim=[2, 3], keepdim=True)  # (batch_size, channel, 1, 1)

        s1 = (2 * feature_mean * target_feature_mean + eps) / (
            feature_mean**2 + target_feature_mean**2 + eps
        )  # (batch_size, ...)
        s2 = (2 * cov + eps) / (feature_var + target_feature_var + eps)  # (batch_size, ...)

        return -(alpha * s1 + beta * s2).mean(dim=tuple(range(1, s1.ndim))) / (
            alpha + beta
        )


class CombinationLoss(Critic):
    """Combination of other Critics"""

    def __init__(
        self, critics: list[Critic], weights: list[float], with_wandb: bool = False
    ) -> None:
        super().__init__(with_wandb=with_wandb)
        self.critics = critics
        self.weights = weights

    def criterion(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        features: dict[str, torch.Tensor],
        target_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for critic, weight in zip(self.critics, self.weights):
            loss += weight * critic(features, target_features)
        if self.with_wandb:
            wandb.log({"critic/loss": loss.mean().item()}, commit=False)
        return loss
