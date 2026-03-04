from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .mamba import BiMambaEncoder

MODEL_NAMES = ["mamba_spectrogram"]


class SpectrogramPatchEmbed(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: tuple[int, int] = (16, 16),
        patch_stride: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)
        x = self.proj(spectrogram)
        x = x.flatten(start_dim=2).transpose(1, 2)
        return self.norm(x)


class MambaSpectrogramClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        patch_size: tuple[int, int] = (16, 16),
        patch_stride: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.embed = SpectrogramPatchEmbed(
            d_model=d_model, patch_size=patch_size, patch_stride=patch_stride
        )
        self.encoder = BiMambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        spectrogram = inputs["spectrogram"] if isinstance(inputs, dict) else inputs
        x = self.embed(spectrogram)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


@dataclass
class ModelConfig:
    model_name: str
    num_classes: int
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    spectrogram_patch_freq: int = 16
    spectrogram_patch_time: int = 16
    spectrogram_stride_freq: int = 16
    spectrogram_stride_time: int = 16


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_name != "mamba_spectrogram":
        raise ValueError(
            f"Unknown model: {config.model_name}. Supported models: {MODEL_NAMES}"
        )
    return MambaSpectrogramClassifier(
        num_classes=config.num_classes,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
        patch_size=(config.spectrogram_patch_freq, config.spectrogram_patch_time),
        patch_stride=(config.spectrogram_stride_freq, config.spectrogram_stride_time),
    )
