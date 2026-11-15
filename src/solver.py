"""Pseudo-spectral Navier–Stokes on a periodic square, vorticity form.

∂ω/∂t + u·∇ω = ν ∇²ω. No forcing. The time step is classical RK4, and the
product in the advection term is dealiased with the 2/3 rule.
"""

from __future__ import annotations

import torch

from src.spectral import SpectralGrid, dealias, derivatives, velocity_from_vorticity


class SpectralNS:
    def __init__(self, n: int, nu: float, dt: float, dtype: torch.dtype = torch.float64):
        self.grid = SpectralGrid(n, dtype=dtype)
        self.nu = float(nu)
        self.dt = float(dt)
        self.dtype = dtype

    def rhs(self, omega: torch.Tensor) -> torch.Tensor:
        u, v = velocity_from_vorticity(omega, self.grid)
        wx, wy, lap = derivatives(omega, self.grid)
        advection = dealias(u * wx + v * wy, self.grid)
        return -advection + self.nu * lap

    def step(self, omega: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dt * k1)
        k3 = self.rhs(omega + 0.5 * dt * k2)
        k4 = self.rhs(omega + dt * k3)
        out = omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return out - out.mean(dim=(-2, -1), keepdim=True)

    def rollout(self, omega0: torch.Tensor, n_steps: int, save_every: int) -> torch.Tensor:
        """Stack the initial field and every `save_every` step. Shape [..., T, n, n]."""
        frames = [omega0]
        omega = omega0
        for i in range(1, n_steps + 1):
            omega = self.step(omega)
            if i % save_every == 0:
                frames.append(omega)
        return torch.stack(frames, dim=-3)


def random_vorticity(n_samples: int, n: int, n_modes: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Smooth, zero-mean vorticity. Each sample is scaled to standard deviation 1."""
    white = torch.randn(n_samples, n, n, generator=generator, dtype=dtype)
    hat = torch.fft.fft2(white)
    axis = torch.fft.fftfreq(n, d=1.0) * n
    kx = axis.view(1, n)
    ky = axis.view(n, 1)
    keep = (kx.abs() <= n_modes) & (ky.abs() <= n_modes)
    keep = keep.clone()
    keep[0, 0] = False
    hat = hat * keep.to(dtype=hat.real.dtype)
    omega = torch.fft.ifft2(hat).real
    omega = omega - omega.mean(dim=(-2, -1), keepdim=True)
    std = omega.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return omega / std
