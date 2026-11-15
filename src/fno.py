"""Fourier neural operator for one step of the vorticity field.

The integral kernel is a learned multiplier on the lowest Fourier modes, so the
same weights run on a finer grid than the one used in training. Absolute (x, y)
channels are left out: the domain is periodic and the equation does not pick
out a preferred point.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes, modes)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if self.modes > height // 2 or self.modes > width // 2:
            raise ValueError(f"{self.modes} modes do not fit in a {height}×{width} grid")
        spectrum = torch.fft.rfft2(x)
        out = torch.zeros(
            batch,
            self.weight_pos.shape[1],
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        modes = self.modes
        out[:, :, :modes, :modes] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, :modes, :modes], self.weight_pos
        )
        out[:, :, -modes:, :modes] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, -modes:, :modes], self.weight_neg
        )
        return torch.fft.irfft2(out, s=(height, width))


class FNO2d(nn.Module):
    def __init__(self, modes: int = 12, width: int = 32, layers: int = 4):
        super().__init__()
        self.modes = modes
        self.lift = nn.Conv2d(1, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        x = self.lift(omega)
        last = len(self.spectral) - 1
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x) + local(x)
            if index < last:
                x = torch.nn.functional.gelu(x)
        out = self.project(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)
