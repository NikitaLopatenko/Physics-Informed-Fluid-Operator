import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver3d import SpectralNS3d
from src.spectral3d import curl, divergence, project


class Solver3dTests(unittest.TestCase):
    def test_projection_kills_divergence(self):
        torch.manual_seed(0)
        grid_n = 16
        solver = SpectralNS3d(grid_n, nu=0.05, dt=0.01)
        raw = torch.randn(2, 3, grid_n, grid_n, grid_n)
        clean = project(raw, solver.grid)
        self.assertLess(float(divergence(clean, solver.grid).abs().max()), 1e-5)

    def test_curl_is_divergence_free(self):
        torch.manual_seed(1)
        grid_n = 16
        solver = SpectralNS3d(grid_n, nu=0.05, dt=0.01)
        potential = torch.randn(2, 3, grid_n, grid_n, grid_n)
        speed = curl(potential, solver.grid)
        self.assertLess(float(divergence(speed, solver.grid).abs().max()), 1e-8)

    def test_abc_flow_decays_at_the_viscous_rate(self):
        # ABC flow is Beltrami: (u·∇)u is a pure gradient and ∇²u = −u, so the
        # projected equation is ∂u/∂t = −ν u.
        n = 16
        nu = 0.1
        dt = 0.01
        t_final = 0.2
        solver = SpectralNS3d(n, nu, dt)
        x, y, z = solver.grid.coords()
        u = torch.sin(z) + torch.cos(y)
        v = torch.sin(x) + torch.cos(z)
        w = torch.sin(y) + torch.cos(x)
        initial = torch.stack([u, v, w], 0)
        frames, _ = solver.rollout(initial, int(t_final / dt), int(t_final / dt))
        expected = initial * math.exp(-nu * t_final)
        self.assertLess(float((frames[-1] - expected).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
