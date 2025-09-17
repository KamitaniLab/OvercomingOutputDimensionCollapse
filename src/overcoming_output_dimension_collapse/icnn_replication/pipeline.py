from __future__ import annotations

import torch
import torch.optim as optim

import wandb

from .critic import Critic
from .encoder import Encoder
from .generator import Generator


class FeatureInversionPipeline:
    """Feature inversion pipeline

    Parameters
    ----------
    generator : Generator
        Generator module that generates images.
    encoder : Encoder
        Encoder module that extracts layer-wise features from images.
    critic : Critic
        Critic module that computes the loss between generated features and target
        features.
    optimizer : optim.Optimizer
        Optimizer module that optimizes the generator.
    scheduler : optim.lr_scheduler.LRScheduler, optional
        Scheduler module that schedules the learning rate of the optimizer, by default
        None.
    num_iterations : int, optional
        Number of iterations, by default 1.
    log_interval : int, optional
        Interval of logging, by default -1. If -1, logging is disabled.
    with_wandb : bool, optional
        Whether to use wandb, by default False.
    """

    def __init__(
        self,
        generator: Generator,
        encoder: Encoder,
        critic: Critic,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        num_iterations: int = 1,
        log_interval: int = -1,
        with_wandb: bool = False,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.encoder = encoder
        self.critic = critic
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_iterations = num_iterations
        self.log_interval = log_interval
        self.with_wandb = with_wandb
        if self.with_wandb:
            self.critic.enable_wandb()

    def __call__(
        self,
        target_features: dict[str, torch.Tensor],
        max_trials: int = 1,
        loss_threshold: float | None = None,
    ) -> torch.Tensor:
        """Forward pass through the iCNN pipeline.

        Parameters
        ----------
        target_features : dict[str, torch.Tensor]
            Target features indexed by the layer names.

        Returns
        -------
        torch.Tensor
            Generated images.
        """
        for trial in range(max_trials):
            self.reset_states()
            for step in range(self.num_iterations):
                self.optimizer.zero_grad()
                generated_images = self.generator()
                generated_features = self.encoder(generated_images)
                loss = self.critic(generated_features, target_features)
                loss.sum().backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.with_wandb:
                    wandb.log({"loss": loss.mean().item()})

                if self.log_interval > 0 and step % self.log_interval == 0:
                    print(
                        f"Step [{step+1}/{self.num_iterations}]: loss={loss.mean().item():.4f}"
                    )
            if loss_threshold is None or loss.mean().item() < loss_threshold:
                break
            else:
                print(
                    f"Loss is not less than {loss_threshold} on trial {trial+1}, retrying..."
                )

        return self.generator().detach()

    def reset_states(self) -> None:
        """Reset the state of the pipeline.

        Notes
        -----
        This method is needed to reset the state of the optimizer and the generator
        when the pipeline is used for multiple stimuli. Otherwise, the initial
        generated image for the second stimulus is the final generated image for the
        first stimulus. Other implementaion idea is to put the optimizer and the
        generator in the __call__ method, instead of the __init__ method.
        """

        self.generator.reset_states()
        self.optimizer = self.optimizer.__class__(
            self.generator.parameters(), **self.optimizer.defaults
        )
