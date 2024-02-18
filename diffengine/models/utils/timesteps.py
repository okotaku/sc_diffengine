import torch
from torch import nn


class WuerstchenRandomTimeSteps(nn.Module):
    """Wuerstchen Random Time Steps module."""

    def forward(self, num_batches: int, device: str,
                ) -> torch.Tensor:
        """Forward pass.

        Generates time steps for the given batches.

        Args:
        ----
            scheduler (DDPMScheduler): Scheduler for training diffusion model.
            num_batches (int): Batch size.
            device (str): Device.

        """
        return torch.rand((num_batches, ), device=device)
