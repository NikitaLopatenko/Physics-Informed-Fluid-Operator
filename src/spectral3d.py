"""3D periodic spectral calculus on [0, 2π]³. Axes are (z, y, x)."""

from __future__ import annotations

import math

import torch


class Grid3d:
    def __init__(self, n: int, dtype: torch.dtype = torch.float64, device: str = "cpu"):
        axis = torch.fft.fftfreq(n, d=2 * math.pi / n, device=device, dtype=dtype) * (2 * math.pi)
        kz = axis.view(n, 1, 1).expand(n, n, n).contiguous()
        ky = axis.view(1, n, 1).expand(n, n, n).contiguous()
        kx = axis.view(1, 1, n).expand(n, n, n).contiguous()
        # The Nyquist bin is its own conjugate. i·k there is not a real derivative, so the multiplier is 0.
        kx[..., n // 2] = 0
        ky[..., n // 2, :] = 0
        kz[n // 2] = 0
        k2 = kx**2 + ky**2 + kz**2
        inv = torch.zeros_like(k2)
        live = k2 > 0
        inv[live] = 1.0 / k2[live]
        dealias = ((kx.abs() <= n / 3) & (ky.abs() <= n / 3) & (kz.abs() <= n / 3)).to(dtype)
        self.n = n
        self.dtype = dtype
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = k2
        self.inv_k2 = inv
        self.dealias = dealias

    def coords(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        axis = torch.arange(self.n, device=self.kx.device, dtype=self.dtype) * (2 * math.pi / self.n)
        z = axis.view(self.n, 1, 1).expand(self.n, self.n, self.n)
        y = axis.view(1, self.n, 1).expand(self.n, self.n, self.n)
        x = axis.view(1, 1, self.n).expand(self.n, self.n, self.n)
        return x, y, z


def _view(grid: Grid3d, field: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    extra = (1,) * (field.ndim - 3)
    return tensor.view(*extra, grid.n, grid.n, grid.n)


def _wave_triple(grid: Grid3d, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _view(grid, field, grid.kx),
        _view(grid, field, grid.ky),
        _view(grid, field, grid.kz),
        _view(grid, field, grid.inv_k2),
    )


def project(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    """Drop the gradient part so the velocity is divergence-free. Shape [..., 3, n, n, n]."""
    hats = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    kx, ky, kz, inv = _wave_triple(grid, velocity)
    div = 1j * (
        kx * hats[..., 0:1, :, :, :] + ky * hats[..., 1:2, :, :, :] + kz * hats[..., 2:3, :, :, :]
    )
    # ∇²φ = ∇·u, so −k² φ̂ = div̂.
    phi = -inv * div
    hats = hats.clone()
    hats[..., 0:1, :, :, :] = hats[..., 0:1, :, :, :] - 1j * kx * phi
    hats[..., 1:2, :, :, :] = hats[..., 1:2, :, :, :] - 1j * ky * phi
    hats[..., 2:3, :, :, :] = hats[..., 2:3, :, :, :] - 1j * kz * phi
    return torch.fft.ifftn(hats, dim=(-3, -2, -1)).real


def divergence(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    hats = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    kx, ky, kz, _ = _wave_triple(grid, velocity)
    div = 1j * (
        kx * hats[..., 0:1, :, :, :] + ky * hats[..., 1:2, :, :, :] + kz * hats[..., 2:3, :, :, :]
    )
    return torch.fft.ifftn(div, dim=(-3, -2, -1)).real.squeeze(-4)


def curl(vector: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    """∇×A. The divergence of this field is zero up to roundoff, for any A."""
    d_ax = gradient(vector[..., 0, :, :, :], grid)
    d_ay = gradient(vector[..., 1, :, :, :], grid)
    d_az = gradient(vector[..., 2, :, :, :], grid)
    # gradient components are (∂/∂x, ∂/∂y, ∂/∂z).
    cx = d_az[..., 1, :, :, :] - d_ay[..., 2, :, :, :]
    cy = d_ax[..., 2, :, :, :] - d_az[..., 0, :, :, :]
    cz = d_ay[..., 0, :, :, :] - d_ax[..., 1, :, :, :]
    return torch.stack((cx, cy, cz), dim=-4)


def gradient(scalar: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    hat = torch.fft.fftn(scalar, dim=(-3, -2, -1))
    kx = _view(grid, scalar, grid.kx)
    ky = _view(grid, scalar, grid.ky)
    kz = _view(grid, scalar, grid.kz)
    comps = (1j * kx * hat, 1j * ky * hat, 1j * kz * hat)
    return torch.stack([torch.fft.ifftn(c, dim=(-3, -2, -1)).real for c in comps], dim=-4)


def laplacian(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    hats = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    k2 = _view(grid, velocity, grid.k2)
    return torch.fft.ifftn(-k2 * hats, dim=(-3, -2, -1)).real


def dealias(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    mask = _view(grid, velocity, grid.dealias)
    return torch.fft.ifftn(torch.fft.fftn(velocity, dim=(-3, -2, -1)) * mask, dim=(-3, -2, -1)).real


def convection(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    """(u·∇)u, dealiased. Components are (u, v, w) = (x, y, z)."""
    pieces = []
    for component in range(3):
        slope = gradient(velocity[..., component, :, :, :], grid)
        pieces.append(
            velocity[..., 0, :, :, :] * slope[..., 0, :, :, :]
            + velocity[..., 1, :, :, :] * slope[..., 1, :, :, :]
            + velocity[..., 2, :, :, :] * slope[..., 2, :, :, :]
        )
    return dealias(torch.stack(pieces, dim=-4), grid)


def pressure_from_velocity(velocity: torch.Tensor, grid: Grid3d) -> torch.Tensor:
    """Poisson pressure ∇²p = −∇·((u·∇)u), mean removed."""
    adv = convection(velocity, grid)
    div = divergence(adv, grid)
    hat = torch.fft.fftn(div, dim=(-3, -2, -1))
    inv = _view(grid, div, grid.inv_k2)
    # ∇·((u·∇)u) has Fourier symbol; ∇²p = -that, so p_hat = div_hat * inv_k2
    # because fft(∇²p) = -k² p_hat = -div_hat ⇒ p_hat = div_hat / k²
    return torch.fft.ifftn(hat * inv, dim=(-3, -2, -1)).real
