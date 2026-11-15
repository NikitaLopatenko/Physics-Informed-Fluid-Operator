"""Pseudo-spectral 3D incompressible Navier–Stokes.

∂u/∂t + (u·∇)u = −∇p + ν∇²u,  ∇·u = 0, periodic box. One RK4 step is four
evaluations of that formula. Pressure is the Lagrange multiplier that keeps
the velocity divergence-free; it is recovered from a Poisson solve when a
frame is saved.
"""

from __future__ import annotations

import torch

from src.spectral3d import Grid3d, convection, laplacian, pressure_from_velocity, project


class SpectralNS3d:
    def __init__(self, n: int, nu: float, dt: float, dtype: torch.dtype = torch.float64):
        self.grid = Grid3d(n, dtype=dtype)
        self.nu = float(nu)
        self.dt = float(dt)

    def rhs(self, velocity: torch.Tensor) -> torch.Tensor:
        adv = convection(velocity, self.grid)
        visc = laplacian(velocity, self.grid)
        return project(-adv + self.nu * visc, self.grid)

    def step(self, velocity: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.rhs(velocity)
        k2 = self.rhs(velocity + 0.5 * dt * k1)
        k3 = self.rhs(velocity + 0.5 * dt * k2)
        k4 = self.rhs(velocity + dt * k3)
        out = velocity + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return project(out, self.grid)

    def rollout(self, velocity: torch.Tensor, n_steps: int, save_every: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return velocity and pressure at the initial time and every save_every step.

        Velocity shape [..., T, 3, n, n, n]. Pressure shape [..., T, n, n, n].
        """
        speeds = [velocity]
        pressures = [pressure_from_velocity(velocity, self.grid)]
        state = velocity
        for i in range(1, n_steps + 1):
            state = self.step(state)
            if i % save_every == 0:
                speeds.append(state)
                pressures.append(pressure_from_velocity(state, self.grid))
        return torch.stack(speeds, dim=-5), torch.stack(pressures, dim=-4)


def random_velocity(n_samples: int, n: int, n_modes: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    white = torch.randn(n_samples, 3, n, n, n, generator=generator, dtype=dtype)
    hats = torch.fft.fftn(white, dim=(-3, -2, -1))
    axis = torch.fft.fftfreq(n, d=1.0) * n
    kz = axis.view(n, 1, 1)
    ky = axis.view(1, n, 1)
    kx = axis.view(1, 1, n)
    keep = (kx.abs() <= n_modes) & (ky.abs() <= n_modes) & (kz.abs() <= n_modes)
    hats = hats * keep.to(dtype=hats.real.dtype)
    field = torch.fft.ifftn(hats, dim=(-3, -2, -1)).real
    field = project(field, Grid3d(n, dtype=dtype))
    scale = field.std(dim=(-4, -3, -2, -1), keepdim=True).clamp(min=1e-8)
    return field / scale
