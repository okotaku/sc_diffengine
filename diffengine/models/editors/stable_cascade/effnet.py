import torch
import torchvision
from huggingface_hub import hf_hub_download
from safetensors.torch import load
from torch import nn


class EfficientNetEncoder(nn.Module):
    """EffcientNetV2 encoder."""

    def __init__(self, c_latent: int = 16) -> None:
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(
            weights="DEFAULT").features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),
        )
        file_path = hf_hub_download(repo_id="stabilityai/stable-cascade",
                              filename="effnet_encoder.safetensors")
        with open(file_path, "rb") as f:  # noqa: PTH123
            data = f.read()

        loaded = load(data)
        self.load_state_dict(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mapper(self.backbone(x))
