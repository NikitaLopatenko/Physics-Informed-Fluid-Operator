"""One forward pass maps the initial 3D velocity to the whole later trajectory.

The output is velocity and pressure at every saved time. Fourier weights sit on
the lowest modes, so the same network runs on a finer mesh than the one it
trained on.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes, modes, modes)
        self.weights = nn.ParameterList(
            nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width = x.shape
        modes = self.modes
        if modes > depth // 2 or modes > height // 2 or modes > width // 2:
            raise ValueError(f"{modes} modes do not fit in a {depth}³ grid")
        spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = torch.zeros(
            batch,
            self.weights[0].shape[1],
            depth,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        slices = (
            (slice(None, modes), slice(None, modes)),
            (slice(-modes, None), slice(None, modes)),
            (slice(None, modes), slice(-modes, None)),
            (slice(-modes, None), slice(-modes, None)),
        )
        for weight, (z_slice, y_slice) in zip(self.weights, slices):
            out[:, :, z_slice, y_slice, :modes] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                spectrum[:, :, z_slice, y_slice, :modes],
                weight,
            )
        return torch.fft.irfftn(out, s=(depth, height, width), dim=(-3, -2, -1))


class FNO3d(nn.Module):
    def __init__(self, modes: int, width: int, layers: int, out_channels: int):
        super().__init__()
        self.modes = modes
        self.out_channels = out_channels
        self.lift = nn.Conv3d(3, width, 1)
        self.spectral = nn.ModuleList(SpectralConv3d(width, width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv3d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv3d(width, width, 1),
            nn.GELU(),
            nn.Conv3d(width, out_channels, 1),
        )

    def forward(self, velocity0: torch.Tensor) -> torch.Tensor:
        x = self.lift(velocity0)
        last = len(self.spectral) - 1
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x) + local(x)
            if index < last:
                x = torch.nn.functional.gelu(x)
        # Each group of 4 channels is a vector potential (Ax, Ay, Az) and a pressure.
        # Velocity is the curl of that potential, applied outside this module.
        return self.project(x)
