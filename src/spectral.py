"""Spectral derivatives for a 2π-periodic square.

Vorticity ω is one number per cell. The velocity comes from a stream function,
∇²ψ = −ω, so the velocity it produces has zero divergence up to roundoff.
A single Fourier mode has no self-advection, because ψ is then proportional to ω.
"""

from __future__ import annotations

import math

import torch


class SpectralGrid:
    def __init__(self, n: int, dtype: torch.dtype = torch.float64, device: str = "cpu"):
        if n < 4:
            raise ValueError("need at least 4 cells on a side")
        axis = torch.fft.fftfreq(n, d=2 * math.pi / n, device=device, dtype=dtype) * (2 * math.pi)
        kx = axis.view(1, n).expand(n, n).contiguous()
        ky = axis.view(n, 1).expand(n, n).contiguous()
        k2 = kx**2 + ky**2
        inv_k2 = torch.zeros_like(k2)
        live = k2 > 0
        inv_k2[live] = 1.0 / k2[live]
        # 2/3 rule: products of modes above n/3 alias back into the kept band.
        dealias = ((kx.abs() <= n / 3) & (ky.abs() <= n / 3)).to(dtype)
        self.n = n
        self.dtype = dtype
        self.device = device
        self.kx = kx
        self.ky = ky
        self.k2 = k2
        self.inv_k2 = inv_k2
        self.dealias = dealias

    def coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        axis = torch.arange(self.n, device=self.device, dtype=self.dtype) * (2 * math.pi / self.n)
        x = axis.view(1, self.n).expand(self.n, self.n)
        y = axis.view(self.n, 1).expand(self.n, self.n)
        return x, y


def _broadcast(grid: SpectralGrid, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extra = (1,) * (field.ndim - 2)
    return (
        grid.kx.view(*extra, grid.n, grid.n),
        grid.ky.view(*extra, grid.n, grid.n),
        grid.inv_k2.view(*extra, grid.n, grid.n),
    )


def velocity_from_vorticity(omega: torch.Tensor, grid: SpectralGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """u = ∂ψ/∂y, v = −∂ψ/∂x, with ∇²ψ = −ω. Mean vorticity is dropped."""
    hat = torch.fft.fft2(omega)
    _, ky, inv_k2 = _broadcast(grid, omega)
    kx, _, _ = _broadcast(grid, omega)
    u = torch.fft.ifft2(1j * ky * inv_k2 * hat).real
    v = torch.fft.ifft2(-1j * kx * inv_k2 * hat).real
    return u, v


def derivatives(omega: torch.Tensor, grid: SpectralGrid) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hat = torch.fft.fft2(omega)
    kx, ky, _ = _broadcast(grid, omega)
    k2 = grid.k2.view(*((1,) * (omega.ndim - 2)), grid.n, grid.n)
    dx = torch.fft.ifft2(1j * kx * hat).real
    dy = torch.fft.ifft2(1j * ky * hat).real
    lap = torch.fft.ifft2(-k2 * hat).real
    return dx, dy, lap


def divergence(omega: torch.Tensor, grid: SpectralGrid) -> torch.Tensor:
    u, v = velocity_from_vorticity(omega, grid)
    hat_u = torch.fft.fft2(u)
    hat_v = torch.fft.fft2(v)
    kx, ky, _ = _broadcast(grid, omega)
    return torch.fft.ifft2(1j * kx * hat_u + 1j * ky * hat_v).real


def dealias(field: torch.Tensor, grid: SpectralGrid) -> torch.Tensor:
    mask = grid.dealias.view(*((1,) * (field.ndim - 2)), grid.n, grid.n)
    return torch.fft.ifft2(torch.fft.fft2(field) * mask).real


def vorticity_residual(
    omega_now: torch.Tensor,
    omega_next: torch.Tensor,
    dt: float,
    nu: float,
    grid: SpectralGrid,
) -> torch.Tensor:
    """∂ω/∂t + u·∇ω − ν∇²ω, with the time slope taken across one saved step.

    Spatial pieces use the average of the two frames, the same centering as
    Crank–Nicolson. Velocity is rebuilt from that average, so divergence is not
    a separate penalty: the stream function already enforces it.
    """
    mid = 0.5 * (omega_now + omega_next)
    u, v = velocity_from_vorticity(mid, grid)
    dx, dy, lap = derivatives(mid, grid)
    tendency = (omega_next - omega_now) / dt
    return tendency + (u * dx + v * dy) - nu * lap
