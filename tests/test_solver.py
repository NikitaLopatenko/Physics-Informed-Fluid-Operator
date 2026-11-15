import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver import SpectralNS
from src.spectral import SpectralGrid, derivatives, divergence, velocity_from_vorticity, vorticity_residual


class SolverTests(unittest.TestCase):
    def test_stream_function_matches_a_two_mode_field(self):
        n = 32
        grid = SpectralGrid(n)
        x, y = grid.coords()
        omega = torch.sin(2 * x) * torch.sin(y)
        u, v = velocity_from_vorticity(omega, grid)
        self.assertLess(float((u - torch.sin(2 * x) * torch.cos(y) / 5).abs().max()), 1e-8)
        self.assertLess(float((v + 2 * torch.cos(2 * x) * torch.sin(y) / 5).abs().max()), 1e-8)
        self.assertLess(float(divergence(omega, grid).abs().max()), 1e-8)

    def test_advection_of_superposed_modes(self):
        n = 64
        grid = SpectralGrid(n)
        x, y = grid.coords()
        omega = torch.sin(x) + torch.sin(2 * x) * torch.sin(y)
        u, v = velocity_from_vorticity(omega, grid)
        wx, wy, lap = derivatives(omega, grid)
        adv = u * wx + v * wy

        u_exact = torch.sin(2 * x) * torch.cos(y) / 5
        v_exact = -torch.cos(x) - 2 * torch.cos(2 * x) * torch.sin(y) / 5
        wx_exact = torch.cos(x) + 2 * torch.cos(2 * x) * torch.sin(y)
        wy_exact = torch.sin(2 * x) * torch.cos(y)
        exact = u_exact * wx_exact + v_exact * wy_exact
        lap_exact = -torch.sin(x) - 5 * torch.sin(2 * x) * torch.sin(y)
        self.assertLess(float((adv - exact).abs().max()), 1e-6)
        self.assertLess(float((lap - lap_exact).abs().max()), 1e-6)

    def test_taylor_green_decays_at_the_viscous_rate(self):
        n = 32
        nu = 0.1
        dt = 0.01
        t_final = 1.0
        solver = SpectralNS(n, nu, dt)
        x, y = solver.grid.coords()
        omega0 = 2 * torch.sin(x) * torch.sin(y)
        frames = solver.rollout(omega0, int(t_final / dt), int(t_final / dt))
        expected = omega0 * math.exp(-2 * nu * t_final)
        self.assertLess(float((frames[-1] - expected).abs().max()), 1e-5)

    def test_saved_solver_step_has_a_smaller_residual_than_doing_nothing(self):
        n = 32
        nu = 1e-2
        dt = 0.01
        solver = SpectralNS(n, nu, dt, dtype=torch.float32)
        generator = torch.Generator().manual_seed(4)
        omega0 = torch.randn(4, n, n, generator=generator, dtype=torch.float32)
        omega0 = omega0 - omega0.mean(dim=(-2, -1), keepdim=True)
        hat = torch.fft.fft2(omega0)
        axis = torch.fft.fftfreq(n) * n
        keep = (axis.abs().view(1, n) <= 4) & (axis.abs().view(n, 1) <= 4)
        omega0 = torch.fft.ifft2(hat * keep).real
        nxt = solver.step(omega0)
        grid = solver.grid
        moved = vorticity_residual(omega0, nxt, dt, nu, grid)
        stayed = vorticity_residual(omega0, omega0, dt, nu, grid)
        self.assertLess(float(moved.pow(2).mean()), float(stayed.pow(2).mean()))


if __name__ == "__main__":
    unittest.main()
